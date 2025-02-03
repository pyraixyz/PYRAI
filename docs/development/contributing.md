# Contributing Guide

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/your-username/pyrai.git
cd pyrai
```

3. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

4. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Workflow

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes

3. Run tests:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run with coverage
pytest --cov=pyrai
```

4. Format code:
```bash
# Format with black
black .

# Sort imports
isort .

# Check types
mypy pyrai
```

5. Commit changes:
```bash
git add .
git commit -m "feat: Add your feature description"
```

6. Push changes:
```bash
git push origin feature/your-feature-name
```

7. Create Pull Request

## Code Style

We follow these conventions:

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings
- [Conventional Commits](https://www.conventionalcommits.org/) for commit messages

### Type Hints

Use type hints for all function arguments and return values:

```python
from typing import Dict, List, Optional

def process_data(
    data: List[float],
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """Process input data.
    
    Args:
        data: Input data to process
        config: Optional configuration
        
    Returns:
        Processed data statistics
    """
    pass
```

### Documentation

- Add docstrings to all public classes and methods
- Include examples in docstrings where appropriate
- Keep documentation up to date with code changes

## Testing

### Test Structure

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Performance tests go in `tests/performance/`

### Writing Tests

```python
import pytest

def test_feature():
    """Test feature functionality."""
    # Arrange
    input_data = [1, 2, 3]
    
    # Act
    result = process_data(input_data)
    
    # Assert
    assert result["mean"] == 2.0
    assert result["std"] > 0
```

### Test Coverage

Aim for 100% test coverage on new code. Use coverage reports to identify gaps:

```bash
pytest --cov=pyrai --cov-report=html
``` 