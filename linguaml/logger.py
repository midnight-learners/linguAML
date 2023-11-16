from loguru import logger
import sys
from pathlib import Path

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{extra[tuner_role]}</magenta> | "
    "<green>Epoch: {extra[epoch]}</green> | "
    "<level>{message}</level>"
)

# Remove the default logger
logger.remove(0)

# Console logger
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
)

# Initialize the extra variables of the logger
logger.configure(
    extra={
        "tuner_role": None,
        "epoch": None
    }
)

def set_log_filepath(logger, filepath: Path | str) -> None:
    """Set the file path of the log file.
    
    Parameters
    ----------
    logger : Logger
        The logger to set the log file path for.
    filepath : Path | str
        The file path of the log file.
    """
    
    # File logger
    logger.add(
        filepath,
        format=LOG_FORMAT,
    )
    