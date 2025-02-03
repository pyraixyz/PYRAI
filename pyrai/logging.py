import os
import sys
import logging
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler

class Logger:
    """PyRAI logging system."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: str = "INFO"
    ):
        """Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Log directory
            level: Logging level
        """
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Create handlers
        self._setup_file_handler(file_formatter)
        self._setup_console_handler(console_formatter)
        
    def _setup_file_handler(self, formatter: logging.Formatter) -> None:
        """Setup file handler."""
        log_file = os.path.join(
            self.log_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def _setup_console_handler(self, formatter: logging.Formatter) -> None:
        """Setup console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
        
    @classmethod
    def get_logger(
        cls,
        name: Optional[str] = None,
        **kwargs
    ) -> "Logger":
        """Get or create logger instance."""
        if name is None:
            name = "pyrai"
        return cls(name, **kwargs) 