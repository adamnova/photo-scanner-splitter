# Custom Instructions for LLM Agents

## Project Overview
This is a Python project for automatically detecting, extracting, and aligning individual photos from scanned images containing multiple photographs. The project uses OpenCV, NumPy, and other image processing libraries.

## Python Version
- **Minimum**: Python 3.8
- **Development**: Python 3.12
- **Target**: Python 3.8+

## Code Quality Standards

### Modern Python Best Practices

#### Type Hints
- **REQUIRED**: Use type hints for all function parameters and return values
- Use `typing` module for complex types (List, Dict, Tuple, Optional, etc.)
- Example:
  ```python
  from typing import List, Optional, Tuple
  
  def detect_photos(self, image_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
      """Detect photos in an image."""
      pass
  ```

#### Code Formatting
- **Formatter**: Use `black` for consistent code formatting
- **Line length**: 100 characters (configured in pyproject.toml)
- **String quotes**: Double quotes preferred by black
- Run before committing: `black photo_splitter/ tests/`

#### Linting
- **Linter**: Use `ruff` for fast, comprehensive linting
- Follow PEP 8 style guidelines
- Fix all linter errors before committing
- Run: `ruff check photo_splitter/ tests/`

#### Type Checking
- **Type Checker**: Use `mypy` for static type checking
- Strict mode enabled for new code
- Run: `mypy photo_splitter/`

#### Docstrings
- **Format**: Google-style docstrings
- **Required for**: All public classes, methods, and functions
- **Example**:
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

### Code Organization

#### Module Structure
- Keep modules focused and single-purpose
- Separate concerns (detection logic, CLI, utilities)
- Use `__init__.py` to define public API

#### Function Design
- Functions should do one thing well
- Keep functions under 50 lines when possible
- Use early returns to reduce nesting
- Prefer composition over inheritance

#### Error Handling
- Use specific exceptions, not bare `except:`
- Provide meaningful error messages
- Log errors appropriately
- Example:
  ```python
  try:
      image = cv2.imread(image_path)
      if image is None:
          raise ValueError(f"Could not read image from {image_path}")
  except Exception as e:
      logger.error(f"Failed to process {image_path}: {e}")
      raise
  ```

### Testing Standards

#### Test Coverage
- Write tests for all new features
- Maintain existing test coverage
- Use unittest framework (consistent with existing tests)
- Test file naming: `test_<module>.py`

#### Test Organization
- Group related tests in test classes
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Use setUp/tearDown for test fixtures
- Clean up resources in tearDown

#### Test Quality
- Test edge cases and error conditions
- Use meaningful assertions
- Avoid test interdependencies
- Keep tests fast and focused

#### Running Tests
```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_detector -v

# Run specific test
python -m unittest tests.test_detector.TestPhotoDetector.test_detect_photos_simple_image -v
```

## CI/CD Pipeline Requirements

### What Must Be Tested in CI
- All unit tests must pass
- Code formatting with black (check mode)
- Linting with ruff (no errors)
- Type checking with mypy (no errors)
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12

### What Should NOT Be in CI
- Interactive features testing (requires user input)
- Visual inspection tests (require manual review)
- Large image processing benchmarks (too slow)
- Platform-specific scanner integrations

## Development Workflow

### Before Committing
1. Format code: `black photo_splitter/ tests/`
2. Check linting: `ruff check photo_splitter/ tests/`
3. Run type checker: `mypy photo_splitter/`
4. Run tests: `python -m unittest discover tests -v`
5. Update documentation if needed

### Adding New Features
1. Write tests first (TDD approach preferred)
2. Implement feature with type hints
3. Add docstrings
4. Run quality checks
5. Update README if user-facing
6. Update CONTRIBUTING.md if developer-facing

### Fixing Bugs
1. Add test that reproduces the bug
2. Fix the bug
3. Ensure test passes
4. Run full test suite
5. Document fix if non-obvious

## Dependencies Management

### Core Dependencies
- opencv-python: Image processing
- numpy: Array operations
- pillow: Image handling
- scikit-image: Advanced image processing

### Development Dependencies
- black: Code formatting
- ruff: Linting
- mypy: Type checking
- pytest: Testing (alternative to unittest)

### Adding Dependencies
- Add to `requirements.txt` for runtime dependencies
- Add to `requirements-dev.txt` for development tools
- Specify minimum versions
- Test with minimum and latest versions

## Security Considerations
- No hardcoded credentials or secrets
- Validate all file paths (prevent directory traversal)
- Sanitize user inputs
- Use secure file operations (check permissions)
- Keep dependencies updated for security patches

## Performance Guidelines
- Optimize hot paths (image processing loops)
- Use NumPy vectorization when possible
- Profile before optimizing
- Document performance trade-offs
- Consider memory usage for large images

## Documentation Standards

### Code Comments
- Use sparingly for complex logic
- Explain "why" not "what"
- Keep comments up to date
- Prefer self-documenting code

### README Updates
- Keep installation instructions current
- Add examples for new features
- Update troubleshooting section
- Include performance tips

### CONTRIBUTING Updates
- Document new development tools
- Update workflow for new requirements
- Add examples for common tasks

## Git Commit Standards

### Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- First line: concise summary (50 chars or less)
- Body: detailed explanation if needed
- Reference issues: "Fixes #123"

### Commit Organization
- One logical change per commit
- Keep commits focused and atomic
- Test each commit independently
- Use meaningful commit messages

## Code Review Checklist
- [ ] All tests pass
- [ ] Code is formatted with black
- [ ] No linting errors from ruff
- [ ] Type hints are present and correct
- [ ] Docstrings are complete and accurate
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] No security issues introduced
- [ ] Performance is acceptable
- [ ] Error handling is robust

## Common Patterns

### File Path Handling
```python
from pathlib import Path

def process_image(self, image_path: Path) -> int:
    """Process an image file."""
    # Use Path objects for cross-platform compatibility
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not image_path.is_file():
        raise ValueError(f"Not a file: {image_path}")
    
    # Process the image
    pass
```

### Configuration Handling
```python
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    """Configuration for photo detector."""
    min_area: int = 10000
    edge_threshold1: int = 50
    edge_threshold2: int = 150
```

### Error Context
```python
try:
    result = process_image(path)
except ValueError as e:
    print(f"Invalid image {path}: {e}")
except Exception as e:
    print(f"Unexpected error processing {path}: {e}")
    raise
```

## Resources
- [PEP 8 – Style Guide for Python Code](https://pep8.org/)
- [PEP 484 – Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [Black Code Style](https://black.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
