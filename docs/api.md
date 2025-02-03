# PyRAI API Reference

## Engine

The main interface for model training and deployment.

```python
class Engine:
    def __init__(self, backend: ModelBackend, config: Config):
        """Initialize engine.
        
        Args:
            backend: Model backend
            config: Configuration
        """
        pass
        
    def train(self, model, train_data, val_data=None, metrics=None):
        """Train model.
        
        Args:
            model: Model to train
            train_data: Training data
            val_data: Validation data
            metrics: Training metrics
            
        Returns:
            Training history
        """
        pass
        
    def deploy(self, model, model_id, version=None):
        """Deploy model.
        
        Args:
            model: Model to deploy
            model_id: Model identifier
            version: Model version
            
        Returns:
            Deployment instance
        """
        pass

## Configuration

Configuration management system.

```python
class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        pass
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        pass
```

## Metrics

Metrics computation system.

```python
class MetricCollection:
    def __init__(self, metrics: Dict[str, Metric]):
        """Initialize metric collection.
        
        Args:
            metrics: Dictionary of metrics
        """
        pass
        
    def update(self, preds: Any, targets: Any):
        """Update metrics.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        pass
        
    def compute(self) -> Dict[str, float]:
        """Compute metrics.
        
        Returns:
            Dictionary of metric values
        """
        pass
```

## Command Line Interface

PyRAI provides several command line tools:

### Model Analysis
```bash
pyrai analyze MODEL_PATH --input-size INPUT_SIZE
```

### Model Export
```bash
pyrai export MODEL_PATH OUTPUT_PATH --format {onnx,torchscript} --input-size INPUT_SIZE
```

### Model Benchmark
```bash
pyrai benchmark MODEL_PATH [--output-dir DIR] [--batch-sizes SIZES] [--iterations N]
```

# PyRai API Documentation

## Authentication

### Login
```http
POST /auth/token
```

Login and get access token.

**Request Body:**
```json
{
    "username": "user@example.com",
    "password": "password123"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer"
}
```

### Refresh Token
```http
POST /auth/refresh
```

Refresh access token.

**Headers:**
- Authorization: Bearer {token}

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer"
}
```

## Models

### List Models
```http
GET /models
```

Get list of all models.

**Headers:**
- Authorization: Bearer {token}

**Query Parameters:**
- skip (optional): Number of records to skip
- limit (optional): Maximum number of records to return

**Response:**
```json
[
    {
        "id": 1,
        "name": "my_model",
        "description": "My test model",
        "model_type": "classification",
        "framework": "pytorch",
        "status": "deployed",
        "versions": [
            {
                "id": 1,
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
    }
]
```

### Create Model
```http
POST /models
```

Create a new model.

**Headers:**
- Authorization: Bearer {token}

**Request Body:**
```json
{
    "name": "new_model",
    "description": "New model description",
    "model_type": "classification",
    "framework": "pytorch",
    "config": {
        "layers": [100, 50, 10],
        "activation": "relu"
    }
}
```

**Response:**
```json
{
    "id": 2,
    "name": "new_model",
    "description": "New model description",
    "model_type": "classification",
    "framework": "pytorch",
    "status": "created",
    "created_at": "2024-01-01T00:00:00Z"
}
```

## Tasks

### List Tasks
```http
GET /tasks
```

Get list of all tasks.

**Headers:**
- Authorization: Bearer {token}

**Query Parameters:**
- skip (optional): Number of records to skip
- limit (optional): Maximum number of records to return

**Response:**
```json
[
    {
        "id": 1,
        "name": "training_task",
        "task_type": "training",
        "status": "running",
        "progress": 45,
        "created_at": "2024-01-01T00:00:00Z"
    }
]
```

### Create Task
```http
POST /tasks
```

Create a new task.

**Headers:**
- Authorization: Bearer {token}

**Request Body:**
```json
{
    "name": "new_task",
    "task_type": "training",
    "params": {
        "epochs": 10,
        "batch_size": 32
    },
    "model_id": 1
}
```

**Response:**
```json
{
    "id": 2,
    "name": "new_task",
    "task_type": "training",
    "status": "pending",
    "created_at": "2024-01-01T00:00:00Z"
}
```

## Monitoring

### Get System Metrics
```http
GET /monitoring/system
```

Get system metrics.

**Headers:**
- Authorization: Bearer {token}

**Response:**
```json
{
    "cpu_usage": 45.2,
    "memory_usage": 1073741824,
    "db_connections": 5,
    "uptime_seconds": 3600,
    "active_tasks": 2
}
```

### Health Check
```http
GET /monitoring/health
```

Check system health.

**Response:**
```json
{
    "status": "healthy",
    "database": true,
    "cache": true,
    "task_queue": true,
    "uptime": 3600
}
``` 