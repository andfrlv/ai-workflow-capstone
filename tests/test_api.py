import pytest
from flask import Flask
from app import create_app, init_logging
import os
from werkzeug.datastructures import FileStorage
import json
from app.api import TEST_TRAINING_DATA_PATH  # Import to reset
from datetime import datetime
import logging


@pytest.fixture
def client(tmp_path):
    app = create_app()
    app.config['TESTING'] = True
    app.config['MODEL_PATH'] = str(tmp_path / 'test_model.pkl')
    app.config['LOG_PATH'] = str(tmp_path / 'test_app.log')
    init_logging(app)
    with app.test_client() as client:
        yield client
    # Robust cleanup
    logger = app.logger
    for handler in logger.handlers[:]:
        handler.flush()
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
    logging.getLogger('revenue_predictor').handlers.clear()


@pytest.fixture
def mock_csv():
    return 'tests/test_data/mock_train.csv'


@pytest.fixture
def invalid_csv():
    return 'tests/test_data/invalid_train.csv'


def test_logs_success(client, mock_csv, tmp_path):
    print(f"test_logs_success: tmp_path = {tmp_path}")
    log_path = tmp_path / 'test_app.log'
    print(f"test_logs_success: log_path = {log_path}")
    with open(mock_csv, 'rb') as f:
        data = {'file': (f, 'mock_train.csv')}
        response = client.post(
            '/train', content_type='multipart/form-data', data=data)
        print(
            f"test_logs_success: /train response status: {response.status_code}")
        assert response.status_code == 200, f"/train failed: {response.json}"
    assert os.path.exists(log_path), f"Log file not created at: {log_path}"
    print(
        f"test_logs_success: Log file size: {os.path.getsize(log_path)} bytes")
    response = client.get('/logs')
    print(f"test_logs_success: /logs response status: {response.status_code}")
    assert response.status_code == 200, f"/logs returned {response.status_code}, expected 200"
    assert response.json['status'] == 'success', f"Expected 'success', got: {response.json}"
    assert 'logs' in response.json, "Response missing 'logs' key"
    assert isinstance(
        response.json['logs'], str), f"Logs not a string: {type(response.json['logs'])}"
    assert "Model trained successfully" in response.json[
        'logs'], "Expected 'Model trained successfully' in logs"


def test_logs_empty(client, tmp_path):
    log_path = tmp_path / 'test_app.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('')
    response = client.get('/logs')
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert response.json['logs'] == "", f"Expected empty log, got: {response.json['logs']}"


def test_logs_no_file(client, tmp_path):
    log_path = tmp_path / 'test_app.log'
    if os.path.exists(log_path):
        logger = client.application.logger
        for handler in logger.handlers[:]:
            handler.flush()
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)
        logging.getLogger('revenue_predictor').handlers.clear()
        try:
            os.remove(log_path)
        except PermissionError as e:
            print(f"test_logs_no_file: Failed to remove log file: {e}")
            raise
    response = client.get('/logs')
    assert response.status_code == 404
    assert response.json['status'] == 'error'
    assert "Log file not found" in response.json['message']


def test_metrics_success(client, mock_csv):
    with open(mock_csv, 'rb') as f:
        data = {'file': (f, 'mock_train.csv')}
        client.post('/train', content_type='multipart/form-data', data=data)
    response = client.get('/metrics')
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert 'metrics' in response.json
    metrics = response.json['metrics']
    assert 'rmse' in metrics
    assert isinstance(metrics['rmse'], float)
    assert metrics['rmse'] >= 0
    assert 'training_date' in metrics
    try:
        datetime.strptime(metrics['training_date'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        assert False, "Invalid training_date format"
    assert 'num_countries' in metrics
    assert isinstance(metrics['num_countries'], int)
    assert metrics['num_countries'] > 0


def test_metrics_no_model(client, tmp_path):
    # Clean up any residual files
    model_path = tmp_path / 'test_model.pkl'
    metrics_path = tmp_path / 'test_model_metrics.json'
    for path in [model_path, metrics_path]:
        if os.path.exists(path):
            os.remove(path)
    # Also clean tests/test_models/ to ensure no interference
    for fname in ['test_model.pkl', 'test_model_metrics.json']:
        residual_path = os.path.join('tests/test_models', fname)
        if os.path.exists(residual_path):
            os.remove(residual_path)
    response = client.get('/metrics')
    assert response.status_code == 404
    assert response.json['status'] == 'error'
    assert 'Model not found' in response.json['message']


def test_predict_all_success(client, mock_csv):
    # Train model to ensure countries are known
    with open(mock_csv, 'rb') as f:
        data = {'file': (f, 'mock_train.csv')}
        client.post('/train', content_type='multipart/form-data', data=data)
        client.application.config['TEST_TRAINING_DATA_PATH'] = mock_csv

    payload = {"date": "2020-01-01"}
    response = client.post('/predict/all',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert response.json['date'] == '2020-01-01'
    assert 'total-revenue' in response.json
    assert isinstance(response.json['total-revenue'], float)
    assert response.json['total-revenue'] >= 0
    assert isinstance(response.json['predictions'], list)
    assert len(response.json['predictions']) > 0
    for pred in response.json['predictions']:
        assert 'country' in pred
        assert 'revenue' in pred
        assert isinstance(pred['revenue'], float)
        assert pred['revenue'] >= 0
    # Verify total-revenue equals sum of predictions
    expected_total = sum(pred['revenue']
                         for pred in response.json['predictions'])
    assert abs(response.json['total-revenue'] - expected_total) < 1e-6


def test_predict_all_invalid_date(client):
    payload = {"date": "2020-13-01"}
    response = client.post('/predict/all',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'Invalid date format' in response.json['message']


def test_predict_all_missing_date(client):
    payload = {}
    response = client.post('/predict/all',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'Missing required field' in response.json['message']


def test_predict_country_success(client, mock_csv):
    # Train model to ensure encoder knows "TestCountry"
    with open(mock_csv, 'rb') as f:
        data = {'file': (f, 'mock_train.csv')}
        client.post('/train', content_type='multipart/form-data', data=data)
        client.application.config['TEST_TRAINING_DATA_PATH'] = mock_csv

    payload = {
        "country": "TestCountry",
        "date": "2020-01-01"
    }
    response = client.post('/predict/country',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert response.json['country'] == 'TestCountry'
    assert response.json['date'] == '2020-01-01'
    assert isinstance(response.json['revenue'], float)
    assert response.json['revenue'] >= 0


def test_predict_country_invalid_country(client, mock_csv):
    with open(mock_csv, 'rb') as f:
        data = {'file': (f, 'mock_train.csv')}
        client.post('/train', content_type='multipart/form-data', data=data)
        client.application.config['TEST_TRAINING_DATA_PATH'] = mock_csv

    payload = {
        "country": "UnknownCountry",
        "date": "2020-01-01"
    }
    response = client.post('/predict/country',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 404
    assert response.json['status'] == 'error'
    assert 'Country not found' in response.json['message']


def test_predict_country_invalid_date(client):
    payload = {
        "country": "TestCountry",
        "date": "2020-13-01"
    }
    response = client.post('/predict/country',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'Invalid date format' in response.json['message']


def test_predict_country_missing_fields(client):
    payload = {
        "country": "TestCountry"
    }
    response = client.post('/predict/country',
                           content_type='application/json',
                           data=json.dumps(payload))
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'Missing required fields' in response.json['message']


def test_train_endpoint_success(client, mock_csv, tmp_path):
    model_path = tmp_path / 'test_model.pkl'
    with open(mock_csv, 'rb') as f:
        data = {'file': (f, 'mock_train.csv')}
        response = client.post(
            '/train', content_type='multipart/form-data', data=data)
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert 'rmse' in response.json
    assert os.path.exists(
        model_path), f"Model file not created at: {model_path}"


def test_train_endpoint_invalid_schema(client, invalid_csv):
    with open(invalid_csv, 'rb') as f:
        data = {'file': (f, 'invalid_train.csv')}
        response = client.post(
            '/train', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'Invalid CSV schema' in response.json['message']


def test_train_endpoint_empty_file(client):
    empty_file = FileStorage(stream=open(
        os.devnull, 'rb'), filename='empty.csv')
    data = {'file': empty_file}
    response = client.post(
        '/train', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert response.json['status'] == 'error'
    assert 'Empty file' in response.json['message']


def test_train_endpoint_large_file(client, tmp_path):
    large_csv = tmp_path / 'large.csv'
    with open(large_csv, 'w') as f:
        f.write(','.join(['year', 'month', 'country',
                'revenue', 'times_viewed', 'date']) + '\n')
        for i in range(10000):
            f.write(f'2020,1,TestCountry,100.0,10.0,2020-01-01\n')
    with open(large_csv, 'rb') as f:
        data = {'file': (f, 'large.csv')}
        response = client.post(
            '/train', content_type='multipart/form-data', data=data)
    assert response.status_code == 200
    assert response.json['status'] == 'success'
