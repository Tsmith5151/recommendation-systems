import logging

FORMAT = "%(asctime)s - %(name)s %(levelname)s - %(message)s"


def get_logger(name: str):
    """Format Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def init_logging():
    """Initialize Logger"""
    logging.basicConfig(format=FORMAT)
