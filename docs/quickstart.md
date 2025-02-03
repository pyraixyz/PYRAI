# Quick Start Guide

## Installation

Install PyRAI using pip:

```bash
pip install pyrai
```

For development installation:

```bash
git clone https://github.com/username/pyrai.git
cd pyrai
pip install -e .[dev]
```

## Basic Usage

### Training a Model

```python
from pyrai import Engine
from pyrai.backends import PyTorchBackend
from pyrai.config import Config

# Initialize engine
config = Config({
    "training": {
        "batch_size": 32,
        "epochs": 10
    },
    "model": {
        "type": "pytorch",
        "device": "cuda"
    }
})

engine = Engine(
    backend=PyTorchBackend(),
    config=config
)

# Train model
history = engine.train(
    model=model,
    train_data=train_data,
    val_data=val_data
)
```

### Model Deployment

```python
# Deploy model
deployment = engine.deploy(
    model=model,
    model_id="my_model",
    version="v1"
)

# Make predictions
predictions = deployment.predict(test_data)
```

### Using Command Line Tools

```bash
# Analyze model
pyrai analyze model.pt --input-size 1,3,224,224

# Export model
pyrai export model.pt model.onnx --format onnx

# Run benchmarks
pyrai benchmark model.pt --batch-sizes 1,8,32
```

## Configuration

PyRAI uses YAML configuration files. Create a custom config:

```yaml
training:
  batch_size: 64
  epochs: 100
  optimizer:
    name: "adamw"
    learning_rate: 0.001
    weight_decay: 0.01

model:
  type: "pytorch"
  device: "cuda"
  save_dir: "models"
```

Load configuration:

```python
from pyrai.config import Config

config = Config.from_yaml("config.yaml")
```

## Next Steps

- Check out the [examples](../examples/) directory
- Read the [API documentation](api.md)
- Learn about [advanced features](advanced.md) 