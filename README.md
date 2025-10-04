# Photo Scanner Splitter

A Python tool to automatically detect, extract, and align individual photos from scanned images containing multiple photographs.

## Features

- **Automatic Photo Detection**: Detects individual photos in scanned images using edge detection and contour analysis
- **Rotation Correction**: Automatically detects and corrects photo rotation for proper alignment
- **Dust and Scratch Removal**: State-of-the-art algorithms to clean up dust, scratches, and film grain from old photos
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

### Enable Dust Removal

Clean up old photos with dust, scratches, and film grain using a quality-optimized algorithm:

```bash
photo-splitter input.jpg -o output_photos --dust-removal
```

This feature uses advanced image processing algorithms optimized for maximum quality:
- Bilateral filtering for edge-preserving noise reduction
- Non-local means denoising with enhanced parameters for superior quality
- Multi-scale morphological operations to detect dust at different sizes
- Adaptive thresholding for precise dust mask creation
- Navier-Stokes inpainting for highest quality restoration

### Advanced Options

```bash
photo-splitter input.jpg -o output_photos --min-area 20000 --dust-removal
```

- `--min-area`: Minimum area (in pixels) for a photo to be detected (default: 10000)
- `--no-rotate`: Disable automatic rotation detection and correction
- `--no-interactive`: Disable interactive preview and confirmation
- `--dust-removal`: Enable dust and scratch removal from photos

## How It Works

1. **Edge Detection**: Uses Canny edge detection to find edges in the scanned image
2. **Contour Finding**: Identifies closed contours that likely represent photo boundaries
3. **Filtering**: Filters contours by minimum area to exclude noise and small artifacts
4. **Extraction**: Extracts each detected photo using its bounding box
5. **Dust Removal** (optional): Applies quality-optimized algorithms to clean dust and scratches:
   - Bilateral filtering for edge-preserving noise reduction
   - Non-local means denoising with enhanced quality parameters
   - Multi-scale morphological operations (3x3, 5x5, 7x7 kernels) to detect dust at different sizes
   - Adaptive thresholding for precise dust detection
   - Navier-Stokes inpainting algorithm for superior quality restoration
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

### Example 4: Restoring Old Photos with Dust

For vintage photos with dust and scratches:

```bash
photo-splitter old_album_scan.jpg -o restored_photos --dust-removal
```

## Tips for Best Results

1. **Scan Quality**: Use high-resolution scans (300 DPI or higher) for better detection
2. **Contrast**: Ensure good contrast between photos and background
3. **Background**: A uniform, light-colored background works best
4. **Spacing**: Leave some space between photos on the scanner bed
5. **Alignment**: Photos don't need to be perfectly aligned - rotation correction will handle small angles
6. **Dust Removal**: For best results with old photos, enable `--dust-removal` to clean up dust spots and scratches with quality-optimized algorithms

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