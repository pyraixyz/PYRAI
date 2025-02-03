import pytest
import json
from pathlib import Path
from engine.monitoring.logger import ModelLogger, ModelMetrics

def test_model_logger(temp_dir):
    """Test model logger."""
    logger = ModelLogger(
        log_dir=str(temp_dir),
        service_name="test-service"
    )
    
    # Test prediction logging
    logger.log_prediction(
        model_id="test_model",
        version_id="v1",
        inputs={"data": [1, 2, 3]},
        outputs={"prediction": 1},
        latency=0.1,
        metadata={"batch_size": 1}
    )
    
    # Test training logging
    logger.log_training(
        model_id="test_model",
        metrics={"accuracy": 0.95},
        parameters={"lr": 0.001},
        duration=100.0
    )
    
    # Test error logging
    logger.log_error(
        error_type="ValidationError",
        error_message="Invalid input",
        context={"input_shape": [1, 2, 3]}
    )

def test_model_metrics():
    """Test model metrics collection."""
    metrics = ModelMetrics(port=8000)
    
    # Test prediction metrics
    metrics.record_prediction(
        model_id="test_model",
        version_id="v1",
        latency=0.1
    )
    
    # Test error metrics
    metrics.record_error(
        model_id="test_model",
        version_id="v1",
        error_type="ValidationError"
    )
    
    # Test memory usage metrics
    metrics.update_memory_usage(
        model_id="test_model",
        version_id="v1",
        memory_bytes=1024 * 1024  # 1MB
    ) 