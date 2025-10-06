# Development Guide

This guide contains detailed information for developers working on the Photo Scanner Splitter project.

## Table of Contents

- [Quick Setup](#quick-setup)
- [Development Tools](#development-tools)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Common Tasks](#common-tasks)

## Quick Setup

1. Clone and install:
```bash
git clone https://github.com/adamnova/photo-scanner-splitter.git
cd photo-scanner-splitter
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

2. Verify setup:
```bash
make test
```

3. (Optional) Install pre-commit hooks:
```bash
make pre-commit-install
```

## Development Tools

### Code Formatting: Black

Black is an opinionated code formatter that ensures consistent code style.

**Configuration**: See `pyproject.toml` under `[tool.black]`

**Usage**:
```bash
# Format all code
make format

# Check formatting without changes
make format-check
```

**Key Settings**:
- Line length: 100 characters
- String quotes: Double quotes
- Target: Python 3.8+

### Linting: Ruff

Ruff is a fast Python linter that combines multiple tools (Flake8, isort, pyupgrade, etc.).

**Configuration**: See `pyproject.toml` under `[tool.ruff]`

**Usage**:
```bash
# Check for issues
make lint-check

# Auto-fix issues
make lint
```

**Enabled Checks**:
- E, W: pycodestyle errors and warnings
- F: Pyflakes (logical errors)
- I: isort (import sorting)
- N: pep8-naming
- UP: pyupgrade (modernize Python code)
- B: flake8-bugbear (common bugs)
- SIM: flake8-simplify

### Type Checking: mypy

mypy performs static type analysis to catch type-related bugs.

**Configuration**: See `pyproject.toml` under `[tool.mypy]`

**Usage**:
```bash
make type-check
```

**Type Hints Required**:
- All function parameters
- All return values
- Complex data structures

**Example**:
```python
from typing import List, Optional, Tuple
import numpy as np

def detect_photos(self, image_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Detect photos in an image."""
    pass
```

### Pre-commit Hooks

Pre-commit hooks run automated checks before each commit.

**Installation**:
```bash
make pre-commit-install
```

**Manual run**:
```bash
make pre-commit-run
```

**Hooks included**:
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON/TOML validation
- Large file detection
- Black formatting
- Ruff linting
- mypy type checking

## Code Quality Standards

### Python Style Guide

Follow PEP 8 with these specific guidelines:

1. **Imports**:
   - Standard library imports first
   - Third-party imports second
   - Local imports last
   - Within each group, alphabetically sorted
   
   ```python
   import os
   import sys
   from pathlib import Path
   
   import cv2
   import numpy as np
   
   from .detector import PhotoDetector
   ```

2. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_CASE`
   - Private: `_leading_underscore`

3. **Docstrings**:
   - Use Google-style docstrings
   - Required for all public classes, methods, functions
   
   ```python
   def extract_photo(self, image: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
       """Extract a single photo from the scanned image.
       
       Args:
           image: The source image containing the photo
           contour: The contour defining the photo boundary
           
       Returns:
           The extracted and cropped photo, or None if extraction fails
           
       Raises:
           ValueError: If the contour is invalid
       """
       pass
   ```

4. **Type Hints**:
   - Always include type hints
   - Use `typing` module for complex types
   - Use `Optional[T]` for nullable values

5. **Error Handling**:
   - Use specific exceptions
   - Provide meaningful error messages
   - Don't use bare `except:`
   
   ```python
   try:
       image = cv2.imread(image_path)
       if image is None:
           raise ValueError(f"Could not read image from {image_path}")
   except Exception as e:
       logger.error(f"Failed to process {image_path}: {e}")
       raise
   ```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m unittest tests.test_detector -v

# Run specific test
python -m unittest tests.test_detector.TestPhotoDetector.test_detect_photos_simple_image -v

# Run with coverage
make test-coverage
```

### Writing Tests

1. **Test Structure**:
   - Place tests in `tests/` directory
   - Name test files `test_<module>.py`
   - Use descriptive test names: `test_<what>_<condition>_<expected>`

2. **Test Organization**:
   ```python
   class TestPhotoDetector(unittest.TestCase):
       """Test cases for PhotoDetector class"""
       
       def setUp(self):
           """Set up test fixtures"""
           self.detector = PhotoDetector()
           
       def tearDown(self):
           """Clean up test files"""
           # Clean up resources
           
       def test_detect_photos_simple_image(self):
           """Test detection on a simple synthetic image"""
           # Test implementation
   ```

3. **Test Coverage**:
   - Aim for >80% code coverage
   - Test edge cases and error conditions
   - Test with various input types and sizes

### Test Fixtures

Use `setUp()` and `tearDown()` for test fixtures:
- Create temporary files/directories in `setUp()`
- Always clean up in `tearDown()`
- Use `tempfile` module for temporary files

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline runs on every push and pull request.

**Workflow file**: `.github/workflows/ci.yml`

**Jobs**:

1. **Code Quality Checks**:
   - Black formatting check
   - Ruff linting
   - mypy type checking (currently non-blocking)

2. **Tests**:
   - Run on Python 3.8, 3.9, 3.10, 3.11, 3.12
   - All tests must pass
   - Package installation verification

3. **Test Coverage**:
   - Generate coverage report
   - Upload to codecov (optional)

### Running CI Checks Locally

Before pushing code, run CI checks locally:

```bash
make ci
```

This runs:
- Format check (black --check)
- Lint check (ruff)
- Tests

### Branch Protection

**The repository requires all CI checks to pass before merging pull requests.**

Branch protection rules enforce:
- All status checks must pass (Code Quality Checks, Tests, Test Coverage)
- At least one code review approval
- Conversations must be resolved

For administrators: See `.github/branch-protection.md` for configuration details.

### Debugging CI Failures

1. **Format failures**:
   ```bash
   make format-check  # See what would change
   make format        # Fix formatting
   ```

2. **Lint failures**:
   ```bash
   make lint-check    # See issues
   make lint          # Auto-fix
   ```

3. **Test failures**:
   ```bash
   make test          # Run tests locally
   python -m unittest tests.test_detector.TestPhotoDetector.test_failing_test -v
   ```

## Common Tasks

### Adding a New Feature

1. Create a branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Write tests first (TDD):
   ```bash
   # Add tests in tests/test_<module>.py
   python -m unittest tests.test_<module>.TestMyFeature -v
   ```

3. Implement feature:
   - Add type hints
   - Add docstrings
   - Follow style guide

4. Run quality checks:
   ```bash
   make all  # Runs format, lint, type-check, test
   ```

5. Commit and push:
   ```bash
   git add .
   git commit -m "Add feature: description"
   git push origin feature/my-feature
   ```

6. Create pull request

### Fixing a Bug

1. Write a test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Run full test suite
5. Submit PR

### Updating Dependencies

1. Update `requirements.txt` or `requirements-dev.txt`
2. Test with new versions:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt --upgrade
   make test
   ```
3. Update version constraints if needed
4. Test on multiple Python versions

### Release Process

1. Update version in `setup.py`
2. Update CHANGELOG (if exists)
3. Run full test suite
4. Tag release:
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

## Makefile Targets

```bash
make help              # Show all available targets
make install           # Install runtime dependencies
make install-dev       # Install development dependencies
make format            # Format code with black
make format-check      # Check formatting
make lint              # Run ruff with auto-fix
make lint-check        # Run ruff without auto-fix
make type-check        # Run mypy
make test              # Run tests
make test-coverage     # Run tests with coverage
make clean             # Clean build artifacts
make all               # Run all quality checks and tests
make ci                # Run CI checks locally
make pre-commit-install # Install pre-commit hooks
make pre-commit-run    # Run pre-commit on all files
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
pip install -e .  # Install package in editable mode
```

### OpenCV Issues

If OpenCV fails to import:
```bash
pip uninstall opencv-python
pip install opencv-python
```

### Type Check Errors

mypy may have issues with OpenCV/NumPy. These are configured to be ignored in `pyproject.toml`.

### Pre-commit Failures

If pre-commit fails:
```bash
pre-commit run --all-files  # See all issues
make format                  # Fix formatting
make lint                    # Fix lint issues
```

## Resources

- [PEP 8 – Style Guide](https://pep8.org/)
- [Black Code Style](https://black.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)

## Getting Help

- Open an issue on GitHub
- Check existing issues and pull requests
- Review CONTRIBUTING.md
