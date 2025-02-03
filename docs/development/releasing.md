# Release Process

## Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Incompatible API changes
- MINOR: New functionality (backwards-compatible)
- PATCH: Bug fixes (backwards-compatible)

## Release Steps

1. Update Version

Update version in `pyrai/__init__.py`:
```python
__version__ = "1.2.3"
```

2. Update Changelog

Add changes to `CHANGELOG.md`:
```markdown
## [1.2.3] - 2024-01-20

### Added
- New feature X
- Support for Y

### Changed
- Improved performance of Z
- Updated dependency requirements

### Fixed
- Bug in feature A
- Issue with B
```

3. Create Release Branch

```bash
git checkout -b release/v1.2.3
```

4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyrai

# Run performance tests
pytest tests/performance/
```

5. Build Documentation

```bash
# Build docs
mkdocs build

# Verify docs
mkdocs serve
```

6. Create Release Commit

```bash
git add pyrai/__init__.py CHANGELOG.md
git commit -m "chore: Release v1.2.3"
```

7. Create Release Tag

```bash
git tag -a v1.2.3 -m "Release v1.2.3"
```

8. Push Release

```bash
git push origin release/v1.2.3
git push origin v1.2.3
```

9. Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/

# Build package
python -m build
```

10. Upload to PyPI

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ pyrai

# Upload to PyPI
twine upload dist/*
```

## Post-Release

1. Merge Release Branch

```bash
git checkout main
git merge release/v1.2.3
git push origin main
```

2. Clean Up

```bash
# Delete release branch
git branch -d release/v1.2.3
git push origin --delete release/v1.2.3
```

3. Update Development Version

Update version in `pyrai/__init__.py`:
```python
__version__ = "1.2.4-dev"
```

## Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Distribution built and tested
- [ ] PyPI upload successful
- [ ] GitHub release created
- [ ] Development version updated 