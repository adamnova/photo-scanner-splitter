# Contributing to Photo Scanner Splitter

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/photo-scanner-splitter.git
   cd photo-scanner-splitter
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. (Optional) Set up pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Format and check your code:
   ```bash
   # Format code with black
   black photo_splitter/ tests/
   
   # Check linting with ruff
   ruff check photo_splitter/ tests/ --fix
   
   # Run type checker (optional but recommended)
   mypy photo_splitter/
   ```

4. Run tests to ensure everything works:
   ```bash
   python -m unittest discover tests -v
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

## Code Quality Tools

### Automated Formatting with Black
Black is an uncompromising code formatter that ensures consistent code style:
```bash
# Format all code
black photo_splitter/ tests/

# Check without modifying
black --check photo_splitter/ tests/
```

### Linting with Ruff
Ruff is a fast Python linter that catches common errors and style issues:
```bash
# Check for issues
ruff check photo_splitter/ tests/

# Auto-fix issues where possible
ruff check photo_splitter/ tests/ --fix
```

### Type Checking with mypy
mypy performs static type checking to catch type-related bugs:
```bash
# Run type checker
mypy photo_splitter/
```

### Pre-commit Hooks
Pre-commit hooks automatically check code quality before each commit:
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Continuous Integration

The project uses GitHub Actions for CI/CD. All pull requests must pass:
- Code formatting checks (black)
- Linting checks (ruff)
- Type checking (mypy)
- Unit tests on Python 3.8, 3.9, 3.10, 3.11, and 3.12
- Test coverage reporting

You can view CI results in the "Actions" tab of the repository.

## Code Guidelines

### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 100)
- Use meaningful variable and function names
- Add type hints to all function parameters and return values
- Add docstrings to all classes and functions (Google style)
- Keep functions focused and single-purpose

### Testing

- Add tests for new features
- Ensure all existing tests pass
- Test with various image types and sizes
- Include edge cases in tests

### Documentation

- Update README.md if adding new features
- Add docstrings to new code
- Update examples if changing CLI behavior

## Areas for Contribution

### Feature Enhancements

- **Better Rotation Detection**: Improve accuracy of rotation angle detection
- **Multiple Detection Algorithms**: Add alternative detection methods
- **Image Quality Enhancement**: Pre-process scans to improve detection
- **Format Support**: Add support for more image formats
- **GUI Interface**: Create a graphical user interface
- **Batch Statistics**: Report statistics about batch processing
- **Face Recognition**: Extend face detection to recognize and group photos by person
- **Face-based Photo Sorting**: Automatically organize photos based on detected people

### Bug Fixes

- Report bugs via GitHub Issues
- Include sample images that demonstrate the issue
- Provide system information (OS, Python version, dependency versions)

### Documentation

- Improve README with more examples
- Add troubleshooting guides
- Create video tutorials
- Translate documentation

### Testing

- Add more test cases
- Test on different operating systems
- Test with various scanner types and qualities

## Project Structure

```
photo-scanner-splitter/
├── photo_splitter/
│   ├── __init__.py
│   ├── detector.py      # Core detection and processing logic
│   └── cli.py          # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_detector.py  # Tests for detector module
│   └── test_cli.py       # Tests for CLI module
├── examples/
│   ├── create_sample.py  # Sample data generator
│   └── README.md         # Examples documentation
├── requirements.txt      # Python dependencies
├── setup.py             # Package configuration
└── README.md            # Main documentation
```

## Key Modules

### detector.py

Contains the `PhotoDetector` class with methods:
- `detect_photos()`: Find photos in a scanned image
- `extract_photo()`: Extract individual photos
- `detect_rotation()`: Detect rotation angle
- `rotate_image()`: Correct rotation
- `detect_faces()`: Detect people in images using deep learning
- `_load_face_detector()`: Load face detection model (internal)

### cli.py

Contains the `PhotoSplitterCLI` class that provides:
- Interactive and non-interactive modes
- Preview functionality
- Batch processing
- User confirmation

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Exact steps to trigger the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Sample Image**: If possible, provide a sample scan that demonstrates the issue
6. **Environment**:
   - Operating System
   - Python version
   - OpenCV version
   - Command used

## Questions?

Feel free to open an issue for questions or suggestions!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
