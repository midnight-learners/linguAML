import logging
from .config import settings

# Console handler
# It streams the logs in the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
# It stores the logs in a file
file_handler = logging.FileHandler(settings.log_filepath, mode="a")
file_handler.setLevel(logging.INFO)

# Logging formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Set formatters for both handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

def get_logger(name: str) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
