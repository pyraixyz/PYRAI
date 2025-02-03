import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import structlog
from opencensus.ext.prometheus import prometheus_metrics
from prometheus_client import start_http_server, Counter, Histogram, Gauge

class ModelLogger:
    """Structured logger for model operations."""
    
    def __init__(self, 
                 log_dir: str,
                 service_name: str,
                 log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            wrapper_class=structlog.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(
            service_name=service_name
        )
        
        # Set log level
        logging.basicConfig(level=getattr(logging, log_level.upper()))
    
    def log_prediction(self,
                      model_id: str,
                      version_id: str,
                      inputs: Dict[str, Any],
                      outputs: Dict[str, Any],
                      latency: float,
                      metadata: Optional[Dict[str, Any]] = None):
        """Log model prediction."""
        self.logger.info(
            "model_prediction",
            model_id=model_id,
            version_id=version_id,
            inputs=inputs,
            outputs=outputs,
            latency=latency,
            **metadata or {}
        )
    
    def log_training(self,
                    model_id: str,
                    metrics: Dict[str, float],
                    parameters: Dict[str, Any],
                    duration: float):
        """Log model training."""
        self.logger.info(
            "model_training",
            model_id=model_id,
            metrics=metrics,
            parameters=parameters,
            duration=duration
        )
    
    def log_error(self,
                 error_type: str,
                 error_message: str,
                 context: Optional[Dict[str, Any]] = None):
        """Log error."""
        self.logger.error(
            "error",
            error_type=error_type,
            error_message=error_message,
            **context or {}
        )

class ModelMetrics:
    """Prometheus metrics for model monitoring."""
    
    def __init__(self, port: int = 8000):
        # Start Prometheus metrics server
        start_http_server(port)
        
        # Define metrics
        self.prediction_count = Counter(
            'model_prediction_count',
            'Number of predictions',
            ['model_id', 'version_id']
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_id', 'version_id']
        )
        
        self.prediction_error_count = Counter(
            'model_prediction_error_count',
            'Number of prediction errors',
            ['model_id', 'version_id', 'error_type']
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage by model',
            ['model_id', 'version_id']
        )
    
    def record_prediction(self,
                        model_id: str,
                        version_id: str,
                        latency: float):
        """Record a prediction."""
        self.prediction_count.labels(
            model_id=model_id,
            version_id=version_id
        ).inc()
        
        self.prediction_latency.labels(
            model_id=model_id,
            version_id=version_id
        ).observe(latency)
    
    def record_error(self,
                    model_id: str,
                    version_id: str,
                    error_type: str):
        """Record a prediction error."""
        self.prediction_error_count.labels(
            model_id=model_id,
            version_id=version_id,
            error_type=error_type
        ).inc()
    
    def update_memory_usage(self,
                          model_id: str,
                          version_id: str,
                          memory_bytes: float):
        """Update model memory usage."""
        self.model_memory_usage.labels(
            model_id=model_id,
            version_id=version_id
        ).set(memory_bytes) 