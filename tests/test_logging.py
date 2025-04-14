import pytest
import logging
import os
from app.logging_config import setup_logging


def test_logging_writes(tmp_path):
    log_path = tmp_path / 'test.log'
    logger = setup_logging(log_path=str(log_path), force=True)
    logger.info("Test message")
    logger.handlers[0].flush()
    logger.handlers[0].close()
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    print(f"test_logging_writes: Log content: {log_content}")
    assert "Test message" in log_content, f"Expected 'Test message' in log, got: {log_content}"


def test_logging_rotation(tmp_path):
    log_path = tmp_path / 'test.log'
    max_bytes = 1000
    logger = setup_logging(log_path=str(log_path),
                           force=True, max_bytes=max_bytes)
    # Write enough to exceed max_bytes
    message = "A" * 300  # Account for timestamp/formatting overhead
    for i in range(6):   # ~1800 bytes + overhead
        logger.info(f"{message} {i}")
    logger.handlers[0].flush()
    logger.handlers[0].close()
    assert os.path.exists(str(log_path) + ".1"), "Log file did not rotate"
    assert os.path.getsize(
        log_path) <= max_bytes, f"File size too large: {os.path.getsize(log_path)}"
    print(
        f"test_logging_rotation: Rotated file exists: {os.path.exists(str(log_path) + '.1')}")
    print(
        f"test_logging_rotation: Final file size: {os.path.getsize(log_path)}")
