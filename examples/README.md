# Examples

This directory contains example scripts and sample images to demonstrate the photo scanner splitter functionality.

## Creating a Sample Scan

Run the `create_sample.py` script to generate a sample scanned image with multiple photos:

```bash
python examples/create_sample.py
```

This will create `sample_scan.jpg` containing 3 photos at different angles.

## Running the Splitter

### Non-Interactive Mode

Process the sample scan automatically:

```bash
photo-splitter examples/sample_scan.jpg -o examples/output --no-interactive
```

### Interactive Mode

Process with preview and confirmation for each photo:

```bash
photo-splitter examples/sample_scan.jpg -o examples/output
```

## Output

The extracted and aligned photos will be saved in the `examples/output/` directory as:
- `sample_scan_photo_1.jpg`
- `sample_scan_photo_2.jpg`
- `sample_scan_photo_3.jpg`

Each photo will be automatically rotated to correct orientation.
