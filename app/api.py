from flask import Blueprint, request, jsonify, current_app
import pandas as pd
from app.model import train_model, predict_country, predict_all_countries, get_model_metrics
from app.config import Config
import os
from datetime import datetime
import tempfile
import logging

api = Blueprint('api', __name__)

# Store training data path for tests
TEST_TRAINING_DATA_PATH = None


@api.route('/logs', methods=['GET'])
def logs_endpoint():
    log_path = current_app.config.get('LOG_PATH', Config.LOG_PATH)
    null_logger = logging.getLogger('null')
    null_logger.handlers = [logging.NullHandler()]
    try:
        if not os.path.exists(log_path):
            null_logger.error(
                f"logs_endpoint: Log file not found at: {log_path}")
            return jsonify({"status": "error", "message": "Log file not found"}), 404
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        null_logger.info("logs_endpoint: Log file retrieved successfully")
        return jsonify({"status": "success", "logs": log_content}), 200
    except Exception as e:
        null_logger.error(f"logs_endpoint: Failed to read log file: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@api.route('/metrics', methods=['GET'])
def metrics_endpoint():
    model_path = current_app.config.get('MODEL_PATH', Config.MODEL_PATH)
    try:
        data_path = current_app.config.get(
            'TEST_TRAINING_DATA_PATH', Config.DATA_PATH) if current_app.config.get('TESTING') else Config.DATA_PATH
        current_app.logger.info(f"Loading training data from: {data_path}")
        training_data = pd.read_csv(data_path)
        training_data['date'] = pd.to_datetime(training_data['date'])

        metrics = get_model_metrics(model_path, training_data)
        current_app.logger.info(
            f"Retrieved metrics: RMSE={metrics['rmse']}, Countries={metrics['num_countries']}")
        return jsonify({
            "status": "success",
            "metrics": metrics
        }), 200
    except FileNotFoundError as e:
        current_app.logger.error(f"Model not found: {str(e)}")
        return jsonify({"status": "error", "message": "Model not found"}), 404
    except Exception as e:
        current_app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@api.route('/predict/all', methods=['POST'])
def predict_all_endpoint():
    if not request.is_json:
        current_app.logger.error("Invalid content type, JSON required")
        return jsonify({"status": "error", "message": "JSON required"}), 400

    data = request.get_json()
    if 'date' not in data:
        current_app.logger.error("Missing required field: date")
        return jsonify({"status": "error", "message": "Missing required field: date"}), 400

    date_str = data['date']
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        current_app.logger.error(f"Invalid date format: {date_str}")
        return jsonify({"status": "error", "message": "Invalid date format, use YYYY-MM-DD"}), 400

    try:
        data_path = current_app.config.get(
            'TEST_TRAINING_DATA_PATH', Config.DATA_PATH) if current_app.config.get('TESTING') else Config.DATA_PATH
        current_app.logger.info(f"Loading training data from: {data_path}")
        training_data = pd.read_csv(data_path)
        training_data['date'] = pd.to_datetime(training_data['date'])

        model_path = current_app.config.get('MODEL_PATH', Config.MODEL_PATH)
        predictions, total_revenue = predict_all_countries(
            date, model_path, training_data)

        current_app.logger.info(
            f"Predicted revenues for {len(predictions)} countries on {date_str}, total: {total_revenue}")
        return jsonify({
            "status": "success",
            "date": date_str,
            "total-revenue": float(total_revenue),
            "predictions": predictions
        }), 200
    except FileNotFoundError as e:
        current_app.logger.error(f"Model or data not found: {str(e)}")
        return jsonify({"status": "error", "message": "Model or data not found"}), 404
    except Exception as e:
        current_app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@api.route('/predict/country', methods=['POST'])
def predict_country_endpoint():
    if not request.is_json:
        current_app.logger.error("Invalid content type, JSON required")
        return jsonify({"status": "error", "message": "JSON required"}), 400

    data = request.get_json()
    required_fields = ['country', 'date']
    if not all(field in data for field in required_fields):
        current_app.logger.error("Missing required fields")
        return jsonify({"status": "error", "message": "Missing required fields: country, date"}), 400

    country = data['country']
    date_str = data['date']

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        current_app.logger.error(f"Invalid date format: {date_str}")
        return jsonify({"status": "error", "message": "Invalid date format, use YYYY-MM-DD"}), 400

    try:
        data_path = current_app.config.get(
            'TEST_TRAINING_DATA_PATH', Config.DATA_PATH) if current_app.config.get('TESTING') else Config.DATA_PATH
        current_app.logger.info(f"Loading training data from: {data_path}")
        training_data = pd.read_csv(data_path)
        training_data['date'] = pd.to_datetime(training_data['date'])
        current_app.logger.info(
            f"Countries: {training_data['country'].unique()}")

        model_path = current_app.config.get('MODEL_PATH', Config.MODEL_PATH)
        revenue = predict_country(country, date, model_path, training_data)

        current_app.logger.info(
            f"Predicted revenue for {country} on {date_str}: {revenue}")
        return jsonify({
            "status": "success",
            "country": country,
            "date": date_str,
            "revenue": float(revenue)
        }), 200
    except FileNotFoundError as e:
        current_app.logger.error(f"Model or data not found: {str(e)}")
        return jsonify({"status": "error", "message": "Model or data not found"}), 404
    except ValueError as e:
        current_app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@api.route('/train', methods=['POST'])
def train():
    temp_path = None
    try:
        if 'file' not in request.files:
            current_app.logger.error("train: No file uploaded")
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            current_app.logger.error("train: Empty file uploaded")
            return jsonify({"status": "error", "message": "Empty file uploaded"}), 400
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size == 0:
            current_app.logger.error("train: Empty file uploaded")
            return jsonify({"status": "error", "message": "Empty file"}), 400
        if file_size > Config.MAX_FILE_SIZE:
            current_app.logger.error(
                f"train: File too large: {file_size} bytes")
            return jsonify({"status": "error", "message": "File too large"}), 400
        file.seek(0)
        temp_path = os.path.join(tempfile.gettempdir(), f"tmp_{file.filename}")
        file.save(temp_path)
        current_app.logger.info(f"train: Saved test CSV to: {temp_path}")
        data = pd.read_csv(temp_path)
        rmse = train_model(data, current_app.config.get(
            'MODEL_PATH', Config.MODEL_PATH))
        current_app.logger.info(
            f"train: Model trained successfully, RMSE: {rmse}")
        return jsonify({"status": "success", "rmse": rmse}), 200
    except ValueError as e:
        current_app.logger.error(f"train: Invalid input: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"train: Training failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                current_app.logger.error(
                    f"train: Failed to remove temp file {temp_path}: {str(e)}")
