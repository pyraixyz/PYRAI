from enum import Enum
from typing import Dict, Any

class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelStatus(str, Enum):
    """Model status enum."""
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    ERROR = "error"

class ModelFramework(str, Enum):
    """Model framework enum."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"

class ModelType(str, Enum):
    """Model type enum."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    EMBEDDING = "embedding"

class CachePrefix(str, Enum):
    """Cache prefix enum."""
    MODEL = "model"
    USER = "user"
    TASK = "task"
    TOKEN = "token"
    RATE_LIMIT = "ratelimit"

# Default configurations
DEFAULT_CONFIG: Dict[str, Any] = {
    "api": {
        "workers": 4,
        "timeout": 60,
        "max_requests": 10000,
    },
    "database": {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
    },
    "cache": {
        "ttl": 3600,
        "max_memory": "1gb",
        "eviction_policy": "allkeys-lru",
    },
    "security": {
        "token_expire": 30,  # minutes
        "password_min_length": 8,
        "max_login_attempts": 5,
        "lockout_time": 300,  # seconds
    },
    "rate_limit": {
        "default": {
            "requests": 100,
            "window": 60,  # seconds
        },
        "auth": {
            "requests": 5,
            "window": 300,  # seconds
        },
    },
    "monitoring": {
        "metrics_interval": 10,  # seconds
        "log_level": "INFO",
        "trace_sample_rate": 0.1,
    },
}

# Error messages
ERROR_MESSAGES = {
    "auth": {
        "invalid_credentials": "Invalid credentials",
        "token_expired": "Token has expired",
        "insufficient_permissions": "Insufficient permissions",
        "account_locked": "Account is locked",
    },
    "validation": {
        "invalid_input": "Invalid input data",
        "missing_field": "Required field is missing",
        "invalid_format": "Invalid data format",
    },
    "rate_limit": {
        "exceeded": "Rate limit exceeded",
        "retry_after": "Please try again after {seconds} seconds",
    },
    "model": {
        "not_found": "Model not found",
        "invalid_state": "Invalid model state",
        "training_failed": "Model training failed",
    },
    "task": {
        "not_found": "Task not found",
        "cancelled": "Task was cancelled",
        "failed": "Task failed: {error}",
    },
}

# HTTP Status Codes
HTTP_200_OK = 200
HTTP_201_CREATED = 201
HTTP_204_NO_CONTENT = 204
HTTP_400_BAD_REQUEST = 400
HTTP_401_UNAUTHORIZED = 401
HTTP_403_FORBIDDEN = 403
HTTP_404_NOT_FOUND = 404
HTTP_409_CONFLICT = 409
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_429_TOO_MANY_REQUESTS = 429
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503 