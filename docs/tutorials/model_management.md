# Model Management Guide

## Model Registry

### Registering Models

```python
from pyrai import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register model
model_id = registry.register_model(
    name="resnet50",
    version="1.0.0",
    architecture={
        "type": "resnet",
        "layers": 50,
        "width_multiplier": 1.0
    },
    metadata={
        "accuracy": 0.76,
        "parameters": 25.6e6,
        "input_shape": [224, 224, 3]
    }
)
```

### Model Versioning

```python
# Create new version
new_version = registry.create_version(
    model_id=model_id,
    version="1.1.0",
    changes={
        "type": "improvement",
        "description": "Improved accuracy by 2%"
    }
)

# List versions
versions = registry.list_versions(model_id)
```

## Model Storage

### Cloud Storage

```python
# Configure storage
registry.configure_storage(
    provider="s3",
    bucket="model-registry",
    region="us-west-2"
)

# Store model
registry.store_model(
    model_id=model_id,
    weights_path="./weights.pt",
    config_path="./config.json"
)
```

### Blockchain Storage

```python
# Store model hash on chain
registry.store_model_hash(
    model_id=model_id,
    chain="solana"
)

# Verify model integrity
is_valid = registry.verify_model(model_id)
```

## Model Deployment

### Container Building

```python
# Build model container
container_id = registry.build_container(
    model_id=model_id,
    base_image="pytorch/pytorch:1.9.0-cuda11.1-cudnn8",
    requirements=["torch", "numpy"]
)
```

### Deployment Configuration

```python
# Configure deployment
deployment = registry.deploy_model(
    model_id=model_id,
    platform="kubernetes",
    replicas=3,
    resources={
        "cpu": "4",
        "memory": "8Gi",
        "gpu": "1"
    },
    scaling={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_cpu_utilization": 80
    }
)
``` 