from flask import Flask
from app.api import api
from app.logging_config import setup_logging
from app.config import Config
import logging


def create_app():
    app = Flask(__name__)
    logging.getLogger('').handlers.clear()
    app.register_blueprint(api)
    init_logging(app)
    return app


def init_logging(app):
    log_path = app.config.get('LOG_PATH', Config.LOG_PATH)
    print(f"init_logging: Config LOG_PATH: {log_path}")
    app.logger = setup_logging(
        log_path, force=app.config.get('TESTING', False))
    app.logger.info(f"init_logging: Using log path: {log_path}")
    app.logger.info(
        f"init_logging: Logger handlers: {[h.baseFilename for h in app.logger.handlers if hasattr(h, 'baseFilename')]}")
