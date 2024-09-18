import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = RotatingFileHandler(log_file, maxBytes=2000000, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Setup loggers
predictor_logger = setup_logger('predictor', 'logs/predictor.log')
fine_tune_logger = setup_logger('fine_tune', 'logs/fine_tune.log')
dataloader_logger = setup_logger('dataloader', 'logs/dataloader.log')