"""PyRAI API package.

This package provides a REST API interface for the PyRAI framework,
allowing external systems to interact with the knowledge distillation
functionality.

Example:
    ```python
    from pyrai.api.server import start_server
    
    # Start API server
    start_server(host="0.0.0.0", port=8000)
    ```
"""

from .server import app, start_server
from .routes import router
from .schemas import (
    ModelConfig,
    DistillationConfigSchema,
    DistillationResponse
)

__all__ = [
    "app",
    "start_server",
    "router",
    "ModelConfig",
    "DistillationConfigSchema",
    "DistillationResponse"
] 