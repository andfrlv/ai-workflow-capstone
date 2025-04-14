import pytest
import pandas as pd
import joblib
import os
from app.model import train_model, load_model, predict_country, predict_all_countries, get_model_metrics
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime


@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'year': [2020, 2020],
        'month': [1, 2],
        'country': ['TestCountry', 'TestCountry'],
        'revenue': [100.0, 200.0],
        'times_viewed': [10.0, 20.0],
        'date': ['2020-01-01', '2020-02-01']
    })


@pytest.fixture
def mock_csv(tmp_path, mock_data):
    csv_path = tmp_path / 'mock_train.csv'
    mock_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_model(tmp_path, mock_data):
    model_path = tmp_path / 'test_model.pkl'
    train_model(mock_data, str(model_path))
    return str(model_path)


def test_get_model_metrics_success(mock_model, mock_data):
    metrics = get_model_metrics(mock_model, mock_data)
    assert isinstance(metrics, dict)
    assert 'rmse' in metrics
    assert isinstance(metrics['rmse'], float)
    assert metrics['rmse'] >= 0
    assert 'training_date' in metrics
    try:
        datetime.strptime(metrics['training_date'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        assert False, "Invalid training_date format"
    assert 'num_countries' in metrics
    assert metrics['num_countries'] == len(mock_data['country'].unique())


def test_get_model_metrics_no_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        get_model_metrics(str(tmp_path / 'nonexistent.pkl'), pd.DataFrame())


def test_predict_all_countries_success(mock_model, mock_data):
    date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    predictions, total_revenue = predict_all_countries(
        date, mock_model, mock_data)
    assert isinstance(predictions, list)
    assert len(predictions) == len(mock_data['country'].unique())
    assert isinstance(total_revenue, float)
    assert total_revenue >= 0
    for pred in predictions:
        assert 'country' in pred
        assert 'revenue' in pred
        assert isinstance(pred['revenue'], float)
        assert pred['revenue'] >= 0
        assert pred['country'] in mock_data['country'].unique()
    expected_total = sum(pred['revenue'] for pred in predictions)
    assert abs(total_revenue - expected_total) < 1e-6


def test_load_model_success(mock_model):
    model = load_model(mock_model)
    assert isinstance(model, RandomForestRegressor)


def test_load_model_predict(mock_model):
    model = load_model(mock_model)
    input_data = pd.DataFrame({
        'country_encoded': [0],
        'month': [1],
        'year': [2020],
        'times_viewed': [10.0],
        'revenue_lag1': [100.0]
    })
    prediction = model.predict(input_data)
    assert len(prediction) == 1
    assert isinstance(prediction[0], float)
    assert prediction[0] >= 0


def test_load_model_missing_file():
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model("nonexistent.pkl")


def test_predict_country_success(mock_model, mock_data):
    date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    revenue = predict_country("TestCountry", date, mock_model, mock_data)
    assert isinstance(revenue, float)
    assert revenue >= 0


def test_predict_country_unknown_country(mock_model, mock_data):
    date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    with pytest.raises(ValueError, match="Country not found"):
        predict_country("UnknownCountry", date, mock_model, mock_data)


def test_train_model_success(mock_data, tmp_path):
    model_path = tmp_path / 'test_model.pkl'
    rmse = train_model(mock_data, str(model_path))
    assert os.path.exists(model_path)
    assert isinstance(rmse, float)
    assert rmse >= 0
    model = joblib.load(model_path)
    assert isinstance(model, RandomForestRegressor)


def test_train_model_invalid_data(tmp_path):
    invalid_data = pd.DataFrame({
        'wrong_column': [1, 2]
    })
    model_path = tmp_path / 'invalid_model.pkl'
    with pytest.raises(ValueError, match="Invalid CSV schema"):
        train_model(invalid_data, str(model_path))


def test_train_model_sparse_data(tmp_path):
    sparse_data = pd.DataFrame({
        'year': [2020],
        'month': [1],
        'country': ['TestCountry'],
        'revenue': [100.0],
        'times_viewed': [10.0],
        'date': ['2020-01-01']
    })
    model_path = tmp_path / 'sparse_model.pkl'
    rmse = train_model(sparse_data, str(model_path))
    assert os.path.exists(model_path)
    assert rmse >= 0
