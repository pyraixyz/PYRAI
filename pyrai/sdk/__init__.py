"""PyRAI SDK package.

This package provides a Python SDK for interacting with PyRAI API
and simplifying knowledge distillation workflows.

Example:
    ```python
    from pyrai.sdk import PyRAIClient, DistillationConfig, ModelConfig
    
    # Initialize client
    client = PyRAIClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    
    # Configure models
    teacher = ModelConfig(
        name="resnet50",
        architecture="resnet",
        parameters={"num_classes": 1000}
    )
    
    student = ModelConfig(
        name="mobilenet",
        architecture="mobilenet",
        parameters={"num_classes": 1000}
    )
    
    # Configure distillation
    config = DistillationConfig(
        method="soft_targets",
        temperature=4.0,
        alpha=0.5
    )
    
    # Create distillation task
    task_id = client.create_distillation(
        teacher_model=teacher,
        student_model=student,
        config=config
    )
    
    # Monitor progress
    result = client.monitor_distillation(task_id)
    print(f"Distillation completed: {result}")
    ```
"""

from .client import PyRAIClient
from .models import (
    DistillationConfig,
    ModelConfig,
    TaskStatus,
    DistillationResult,
    DistillationMethod
)
from .exceptions import PyRAIError, APIError
from .utils import load_model_config, export_torch_model, export_tf_model

__all__ = [
    "PyRAIClient",
    "DistillationConfig",
    "ModelConfig",
    "TaskStatus",
    "DistillationResult",
    "DistillationMethod",
    "PyRAIError",
    "APIError",
    "load_model_config",
    "export_torch_model",
    "export_tf_model"
] 