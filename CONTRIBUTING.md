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
   pip install -e .
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests to ensure everything works:
   ```bash
   python -m unittest discover tests -v
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Code Guidelines

### Python Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all classes and functions
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
