# Quick Start Guide

## Basic Usage

### Initialize Engine

```python
from pyrai import Engine
from pyrai.backends import PyTorchBackend

# Create engine instance
engine = Engine(
    backend=PyTorchBackend(),
    config={
        "training": {
            "batch_size": 32,
            "epochs": 10
        }
    }
)
```

### Train Model

```python
# Define or load your model
model = MyModel()

# Prepare data
train_data = MyDataset()

# Train model
history = engine.train(
    model=model,
    train_data=train_data,
    validation_data=val_data
)
```

### Deploy Model

```python
# Deploy trained model
deployment = engine.deploy(
    model_id="my_model",
    version="v1",
    config={
        "resources": {
            "cpu": "1",
            "memory": "2Gi"
        },
        "scaling": {
            "min_replicas": 1,
            "max_replicas": 3
        }
    }
)
```

### Monitor Performance

```python
# Get model metrics
metrics = engine.monitor.get_metrics(
    model_id="my_model",
    version="v1",
    metric_names=["latency", "throughput"]
)

# Configure alerts
engine.monitor.set_alert(
    name="high_latency",
    condition="latency > 100ms",
    actions=["notify_slack"]
)
```

## Complete Example

Here's a complete example using PyTorch:

```python
import torch
from pyrai import Engine
from pyrai.backends import PyTorchBackend

# Define model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

# Create data
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

# Initialize engine
engine = Engine(
    backend=PyTorchBackend(),
    config={
        "training": {
            "batch_size": 32,
            "epochs": 5,
            "optimizer": {
                "type": "adam",
                "lr": 0.001
            }
        }
    }
)

# Train model
model = SimpleModel()
history = engine.train(model, X, y)

# Deploy model
deployment = engine.deploy(
    model_id="simple_model",
    version="v1"
)

# Make predictions
predictions = deployment.predict(
    torch.randn(1, 10)
)
``` 