# Development Guide

## Setup Development Environment

1. Clone repository:
```bash
git clone https://github.com/username/pyrai.git
cd pyrai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install development dependencies:
```bash
pip install -e .[dev]
```

## Code Style

We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run formatters:
```bash
black .
isort .
```

Run linters:
```bash
flake8 .
mypy pyrai
```

## Testing

Run tests:
```bash
pytest tests/

# With coverage
pytest --cov=pyrai tests/
```

## Documentation

Build documentation:
```bash
cd docs
make html
```

View documentation:
```bash
python -m http.server -d _build/html
```

## Pull Request Process

1. Create feature branch:
```bash
git checkout -b feature/your-feature
```

2. Make changes and commit:
```bash
git add .
git commit -m "Add your feature"
```

3. Run tests and linting
4. Update documentation if needed
5. Push changes and create PR:
```bash
git push origin feature/your-feature
```

## Release Process

1. Update version in:
   - setup.py
   - docs/conf.py
   - CHANGELOG.md

2. Create release commit:
```bash
git commit -m "Release v0.1.0"
```

3. Tag release:
```bash
git tag v0.1.0
```

4. Push to GitHub:
```bash
git push origin main --tags
```

5. Create GitHub release
6. Publish to PyPI:
```bash
python -m build
twine upload dist/*
``` 