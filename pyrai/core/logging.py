import logging
import logging.config
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import uuid

import structlog
from pythonjsonlogger import jsonlogger

from pyrai.api.config import settings
from pyrai.core.config import config

def setup_logging() -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Load logging config
    with open("config/logging.yml", "r") as f:
        logging_config = yaml.safe_load(f)
    
    # Configure logging
    logging.config.dictConfig(logging_config)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for logging."""
    
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any]
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add environment
        log_record['environment'] = config.app.environment
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id

def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)

class LoggerMiddleware:
    """Middleware for adding request context to logs."""
    
    async def __call__(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and add context to logs."""
        # Add request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
        # Add correlation ID
        correlation_id = request.headers.get('X-Correlation-ID', request_id)
        
        # Add context to structlog
        logger = get_logger()
        with structlog.contextvars.bound_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method
        ):
            # Log request
            logger.info(
                "request_started",
                path=request.url.path,
                method=request.method
            )
            
            try:
                # Process request
                response = await call_next(request)
                
                # Log response
                logger.info(
                    "request_completed",
                    status_code=response.status_code
                )
                return response
            except Exception as e:
                # Log error
                logger.exception(
                    "request_failed",
                    error=str(e)
                )
                raise

logging.setLoggerClass(CustomLogger)
logger = logging.getLogger("pyrai") 