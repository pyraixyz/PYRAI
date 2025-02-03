# Installation Guide

## Prerequisites

Before installing PyRAI, ensure you have:

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation Methods

### From PyPI

```bash
pip install pyrai
```

### From Source

```bash
git clone https://github.com/username/pyrai.git
cd pyrai
pip install -e .
```

## Optional Dependencies

### GPU Support

For GPU support, install CUDA dependencies:

```bash
pip install pyrai[gpu]
```

### Distributed Training

For distributed training support:

```bash
pip install pyrai[distributed]
```

### All Features

To install all optional dependencies:

```bash
pip install pyrai[all]
```

## Verification

Verify the installation:

```python
import pyrai
print(pyrai.__version__)
```

## Development Setup

For development:

```bash
# Clone repository
git clone https://github.com/username/pyrai.git
cd pyrai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
``` 