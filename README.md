# PyRAI

[![CI](https://github.com/username/pyrai/workflows/CI/badge.svg)](https://github.com/username/pyrai/actions)
[![PyPI version](https://badge.fury.io/py/pyrai.svg)](https://badge.fury.io/py/pyrai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyRAI is a flexible and extensible deep learning framework that supports multiple backends (PyTorch and TensorFlow).

## Features

- Multiple backend support (PyTorch, TensorFlow)
- Unified model training and deployment interface
- Comprehensive monitoring and logging
- Flexible configuration system
- Model version management
- Data processing utilities
- Metric computation and tracking

## Installation

```bash
pip install pyrai
```

## Quick Start

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

# Deploy model
deployment = engine.deploy(model, "my_model")

# Make predictions
predictions = deployment.predict(test_data)
```

## Documentation

For detailed documentation, visit [docs/](docs/).

## Examples

Check out our example implementations:

- [MNIST Classification](examples/mnist_example.py)
- [CIFAR-10 Classification](examples/image_classification.py)
- [AG News Text Classification](examples/text_classification.py)

## Command Line Tools

PyRAI provides several command line tools:

```bash
# Analyze model
pyrai analyze model.pt --input-size 1,3,224,224

# Export model
pyrai export model.pt model.onnx --format onnx --input-size 1,3,224,224

# Run benchmarks
pyrai benchmark model.pt --batch-sizes 1,8,32 --iterations 100
```

## Development

```bash
# Clone repository
git clone https://github.com/username/pyrai.git
cd pyrai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run linting
black .
isort .
flake8 .
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Community

- [Discord](https://discord.gg/pyrai)
- [Twitter](https://twitter.com/pyraiXYZ)
- [GitHub Discussions](https://github.com/username/pyrai/discussions) 