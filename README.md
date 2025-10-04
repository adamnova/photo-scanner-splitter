# Photo Scanner Splitter

A Python tool to automatically detect, extract, and align individual photos from scanned images containing multiple photographs.

## Features

- **Automatic Photo Detection**: Detects individual photos in scanned images using edge detection and contour analysis
- **High-Quality Face Detection**: Detects people in images using deep learning-based face detection with ResNet SSD model
- **Rotation Correction**: Automatically detects and corrects photo rotation for proper alignment
- **Interactive Validation**: Preview detected photos and confirm before saving
- **Batch Processing**: Process single images or entire directories
- **Flexible Configuration**: Customize detection sensitivity and processing options

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install from source

```bash
git clone https://github.com/adamnova/photo-scanner-splitter.git
cd photo-scanner-splitter
pip install -r requirements.txt
pip install -e .
```

## Usage

### Basic Usage

Process a single scanned image:

```bash
photo-splitter input.jpg -o output_photos
```

This will:
1. Detect all photos in `input.jpg`
2. Show a preview of detected photos with bounding boxes
3. For each photo, show a preview and ask for confirmation
4. Save accepted photos to `output_photos/` directory

### Batch Processing

Process all images in a directory:

```bash
photo-splitter scans/ -o output_photos
```

### Non-Interactive Mode

Process images without user interaction (useful for batch processing):

```bash
photo-splitter scans/ -o output_photos --no-interactive
```

### Disable Auto-Rotation

If you want to keep the original orientation:

```bash
photo-splitter input.jpg -o output_photos --no-rotate
```

### Advanced Options

```bash
photo-splitter input.jpg -o output_photos --min-area 20000
```

- `--min-area`: Minimum area (in pixels) for a photo to be detected (default: 10000)
- `--no-rotate`: Disable automatic rotation detection and correction
- `--no-interactive`: Disable interactive preview and confirmation

## Face Detection

The tool includes high-quality face detection powered by a deep learning ResNet SSD model. This can be used to detect people in images with high accuracy.

### Using Face Detection Programmatically

```python
from photo_splitter.detector import PhotoDetector
import cv2

# Initialize detector with custom face confidence threshold
detector = PhotoDetector(face_confidence=0.5)

# Load your image
image = cv2.imread('your_photo.jpg')

# Detect faces
faces = detector.detect_faces(image)

# Process results
for i, face in enumerate(faces, 1):
    x, y, w, h = face['bbox']
    confidence = face['confidence']
    print(f"Face {i}: Location ({x}, {y}) Size: {w}x{h} Confidence: {confidence:.2%}")
    
    # Draw rectangle around face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save annotated image
cv2.imwrite('faces_detected.jpg', image)
```

### Face Detection Parameters

- **face_confidence**: Minimum confidence threshold (0.0 to 1.0, default: 0.5)
  - Lower values detect more faces but may include false positives
  - Higher values are more strict and only detect high-confidence faces

### Face Detection Model

The face detector uses OpenCV's DNN module with a pre-trained ResNet SSD model:
- **Model**: res10_300x300_ssd_iter_140000.caffemodel
- **Architecture**: Single Shot Detector (SSD) with ResNet-10 backbone
- **Input Size**: 300x300 pixels
- **Accuracy**: High accuracy on frontal and near-frontal faces

The model is automatically downloaded on first use and cached in `~/.photo_splitter/`.

### Example Demo

Run the included face detection demo:

```bash
python examples/face_detection_demo.py
```

This will create a sample image with face-like patterns, detect them, and save an annotated result.

## How It Works

1. **Edge Detection**: Uses Canny edge detection to find edges in the scanned image
2. **Contour Finding**: Identifies closed contours that likely represent photo boundaries
3. **Filtering**: Filters contours by minimum area to exclude noise and small artifacts
4. **Extraction**: Extracts each detected photo using its bounding box
5. **Rotation Detection**: Analyzes edges to determine if the photo is rotated
6. **Alignment**: Rotates the photo to correct orientation if needed
7. **User Validation**: (if interactive mode) Shows preview for user confirmation
8. **Saving**: Saves the processed photo with a descriptive filename

## Examples

### Example 1: Family Photo Album Scan

You have scanned pages from a photo album where each scan contains 2-4 photos:

```bash
photo-splitter album_scans/ -o individual_photos
```

The tool will:
- Process each scan in the `album_scans/` directory
- Detect individual photos on each page
- Correct any rotation
- Save them as `scan1_photo_1.jpg`, `scan1_photo_2.jpg`, etc.

### Example 2: Quick Batch Processing

You have many scans and want to process them all without manual review:

```bash
photo-splitter large_scan_batch/ -o photos --no-interactive
```

### Example 3: Scanning Larger Photos

If you're scanning larger photos and the default minimum area is too small:

```bash
photo-splitter input.jpg -o output --min-area 50000
```

## Tips for Best Results

1. **Scan Quality**: Use high-resolution scans (300 DPI or higher) for better detection
2. **Contrast**: Ensure good contrast between photos and background
3. **Background**: A uniform, light-colored background works best
4. **Spacing**: Leave some space between photos on the scanner bed
5. **Alignment**: Photos don't need to be perfectly aligned - rotation correction will handle small angles

## Troubleshooting

### Photos not detected

- Try lowering the `--min-area` threshold
- Ensure there's good contrast between photos and background
- Check that photos aren't touching each other in the scan

### Incorrect rotation

- The rotation detection works best with photos that have clear horizontal or vertical edges
- For photos with mostly diagonal content, manual rotation may be needed

### Too many false detections

- Increase the `--min-area` threshold
- Ensure the scanner bed is clean

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Project Structure

```
photo-scanner-splitter/
├── photo_splitter/
│   ├── __init__.py
│   ├── detector.py      # Core photo detection and processing
│   └── cli.py          # Command-line interface
├── tests/              # Test files
├── examples/           # Example images and usage
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
└── README.md          # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses OpenCV for image processing
- Built with Python and NumPy