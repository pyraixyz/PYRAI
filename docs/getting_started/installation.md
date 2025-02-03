# Installation Guide

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Solana CLI tools

## Installation Methods

### From PyPI

```bash
pip install pyrai
```

### From Source

```bash
git clone https://github.com/pyraixyz/pyrai.git
cd pyrai
pip install -e ".[dev]"
```

### Docker

```bash
docker pull pyraixyz/pyrai:latest
docker run -it pyraixyz/pyrai:latest
```

## Verify Installation

```python
import pyrai
print(pyrai.__version__)
``` 