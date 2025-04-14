import os


class Config:
    DATA_PATH = 'data/processed_revenue_data.csv'
    MODEL_PATH = 'models/random_forest_model.pkl'
    TEST_MODEL_PATH = 'tests/test_models/test_model.pkl'
    LOG_PATH = 'logs/app.log'
    TEST_LOG_PATH = 'tests/test_logs/test.log'
    MAX_FILE_SIZE = 10 * 1024 * 1024  # file size limit for scale (bytes)
    MAX_LOG_SIZE = 1024 * 1024  # 1MB
