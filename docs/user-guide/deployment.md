# Deployment Guide

## Model Deployment

### Basic Deployment

```python
from pyrai import Engine
from pyrai.backends import PyTorchBackend

# Initialize engine
engine = Engine(backend=PyTorchBackend())

# Deploy model
deployment = engine.deploy(
    model_id="my_model",
    version="v1",
    config={
        "resources": {
            "cpu": "1",
            "memory": "2Gi"
        }
    }
)

# Get prediction endpoint
endpoint = deployment.endpoint
print(f"Model endpoint: {endpoint}")
```

### Advanced Configuration

```python
deployment_config = {
    "resources": {
        "cpu": "1",
        "memory": "2Gi",
        "gpu": "1"
    },
    "scaling": {
        "min_replicas": 1,
        "max_replicas": 3,
        "target_cpu_utilization": 80,
        "target_memory_utilization": 80
    },
    "networking": {
        "port": 8000,
        "protocol": "http",
        "path": "/predict"
    },
    "security": {
        "authentication": True,
        "ssl_enabled": True
    }
}

deployment = engine.deploy(
    model_id="my_model",
    version="v1",
    config=deployment_config
)
```

## Serving Models

### REST API

```python
# Make predictions via REST API
import requests

response = requests.post(
    f"{deployment.endpoint}/predict",
    json={
        "inputs": [[1.0, 2.0, 3.0]]
    }
)

predictions = response.json()["predictions"]
```

### gRPC Service

```python
from pyrai.client import GRPCClient

# Create gRPC client
client = GRPCClient(deployment.grpc_endpoint)

# Make predictions
predictions = client.predict(
    inputs=[[1.0, 2.0, 3.0]]
)
```

## Deployment Management

### Scaling

```python
# Scale deployment
deployment.scale(replicas=3)

# Auto-scaling configuration
deployment.configure_autoscaling(
    min_replicas=1,
    max_replicas=5,
    target_cpu_utilization=80
)
```

### Updates

```python
# Update deployment
deployment.update(
    resources={
        "cpu": "2",
        "memory": "4Gi"
    }
)

# Rolling update
deployment.rolling_update(
    new_version="v2",
    strategy={
        "max_surge": 1,
        "max_unavailable": 0
    }
)
```

### Health Checks

```python
# Configure health checks
deployment.configure_health_checks(
    readiness_probe={
        "http_get": {
            "path": "/health",
            "port": 8000
        },
        "initial_delay_seconds": 10,
        "period_seconds": 30
    }
)

# Get deployment status
status = deployment.get_status()
print(f"Deployment status: {status}")
``` 