"""
Centralized logging configuration for the climate policy extractor.
"""
import logging
# import colorlog # No longer needed for simplified version
# from scrapy.utils.project import get_project_settings # No longer needed for simplified version

# def setup_colored_logging(logger=None):
#     pass # Removed content

def get_logger(name):
    """
    Get a standard logger.
    """
    logger = logging.getLogger(name)

    # Set default level if not already set
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    # Add console output if no handlers present on this logger
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)

    return logger
