"""Logging utilities for the climate tracker project.

This module provides custom logging formatters and configuration
to enhance logging output with colors and consistent formatting.
"""

import logging
import sys

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages.
    
    This formatter adds ANSI color codes to log messages based on their
    severity level, making it easier to visually distinguish between
    different types of logs.
    
    Colors used:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Bold Red
    """
    
    # More visible ANSI color codes
    cyan = "\033[36m"
    green = "\033[32m"
    yellow = "\033[33m" 
    red = "\033[31m"
    bold_red = "\033[1;31m"
    reset = "\033[0m"
    
    FORMATS = {
        logging.DEBUG: cyan + "%(asctime)s [%(name)s] %(levelname)s: %(message)s" + reset,
        logging.INFO: green + "%(asctime)s [%(name)s] %(levelname)s: %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s [%(name)s] %(levelname)s: %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s [%(name)s] %(levelname)s: %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s [%(name)s] %(levelname)s: %(message)s" + reset
    }

    def __init__(self):
        """Initialize with Scrapy's date format."""
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        """Format the log record with appropriate color.
        
        Args:
            record: The log record to format
            
        Returns:
            str: The formatted log message with color codes
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)

def setup_colored_logging(logger):
    """Configure a logger with colored output to console."""
    # Create console handler that writes to stderr (to separate from Scrapy's stdout)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ColorFormatter())
    
    # Add our handler to the logger
    logger.addHandler(console_handler)
    logger.propagate = False  # Prevent propagation to avoid duplicate logs 