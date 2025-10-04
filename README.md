# Photo Scanner Splitter

A Python tool to automatically detect, extract, and align individual photos from scanned images containing multiple photographs.

## Features

- **Automatic Photo Detection**: Detects individual photos in scanned images using edge detection and contour analysis
- **Rotation Correction**: Automatically detects and corrects photo rotation for proper alignment
- **Duplicate Detection**: Optionally skip duplicate photos using perceptual hashing
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

### Enable Duplicate Detection

When processing multiple scans that may contain the same photos, enable deduplication to automatically skip duplicates:

```bash
photo-splitter scans/ -o output_photos --enable-dedup
```

This feature uses perceptual hashing to detect visually similar images. When a duplicate is detected, it will be skipped and not saved again. This is particularly useful when:
- Processing multiple scans that may overlap
- Re-scanning photos for better quality
- Organizing large photo collections

### Advanced Options

```bash
photo-splitter input.jpg -o output_photos --min-area 20000
```

- `--min-area`: Minimum area (in pixels) for a photo to be detected (default: 10000)
- `--no-rotate`: Disable automatic rotation detection and correction
- `--no-interactive`: Disable interactive preview and confirmation
- `--enable-dedup`: Enable duplicate detection to skip saving duplicate photos

## How It Works

1. **Edge Detection**: Uses Canny edge detection to find edges in the scanned image
2. **Contour Finding**: Identifies closed contours that likely represent photo boundaries
3. **Filtering**: Filters contours by minimum area to exclude noise and small artifacts
4. **Extraction**: Extracts each detected photo using its bounding box
5. **Duplicate Detection**: (if enabled) Computes perceptual hash and checks against previously processed photos
6. **Rotation Detection**: Analyzes edges to determine if the photo is rotated
7. **Alignment**: Rotates the photo to correct orientation if needed
8. **User Validation**: (if interactive mode) Shows preview for user confirmation
9. **Saving**: Saves the processed photo with a descriptive filename

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

### Example 4: Batch Processing with Deduplication

Processing multiple scans that might contain duplicate photos:

```bash
photo-splitter multiple_scans/ -o all_photos --enable-dedup --no-interactive
```

This will:
- Process all scans in the `multiple_scans/` directory
- Automatically skip duplicate photos across all scans
- Save only unique photos to `all_photos/`

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
- Uses ImageHash library for perceptual hashing and duplicate detection