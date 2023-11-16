from pathlib import Path
import logging
from xpyutils import singleton
from .config import settings

LOGGER_NAME = "lingual"

@singleton
class Logger(logging.Logger):
    
    def __init__(
            self,
            log_filepath: Path | str = settings.log_filepath
        ) -> None:
        
        # Initialize super class
        super().__init__(LOGGER_NAME, logging.INFO)
        
        # Console handler
        # It streams the logs in the terminal
        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(logging.INFO)
        
        # File handler
        # It stores the logs in a file
        self._log_filepath = Path(log_filepath)
        self._file_handler = logging.FileHandler(self._log_filepath, mode="a")
        self._file_handler.setLevel(logging.INFO)
        
        # Logging formatter
        self._formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
        )
        
        # Set formatters for both handlers
        self._console_handler.setFormatter(self._formatter)
        self._file_handler.setFormatter(self._formatter)
        
        # Add handlers
        self.addHandler(self._console_handler)
        self.addHandler(self._file_handler)
        
        # Extra information
        self._extra = {}
    
    @property
    def log_filepath(self) -> Path:
        """Log file path.
        """
        
        return self._log_filepath
    
    @log_filepath.setter
    def log_filepath(self, new_filepath: Path | str) -> None:
        
        # The new log file path
        self._log_filepath = Path(new_filepath)
        
        # Close the file handler
        self._file_handler.close()
        
        # Remove the old file handler
        self.removeHandler(self._file_handler)
        
        # Set the new file handler
        self._file_handler = logging.FileHandler(self._log_filepath, mode="a")
        self._file_handler.setLevel(logging.INFO)
        self._file_handler.setFormatter(self._formatter)
        self.addHandler(self._file_handler)
        
    @property        
    def extra(self) -> dict:
        """Extra information provided to the logger.
        """
        
        return self._extra

logger = Logger()
