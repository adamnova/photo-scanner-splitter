# Quick Start Guide

Get started with Photo Scanner Splitter in just a few minutes!

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adamnova/photo-scanner-splitter.git
cd photo-scanner-splitter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Your First Scan

### Step 1: Try the Example

Run our example to see how it works:

```bash
# Create a sample scan
python examples/create_sample.py

# Process it (non-interactive)
photo-splitter examples/sample_scan.jpg -o examples/output --no-interactive
```

You should see output like:
```
Found 1 image(s) to process
Processing: sample_scan.jpg
  Detected 3 photo(s)
  Photo 1: Detected rotation of -7.0°
  Saved: sample_scan_photo_1.jpg
  ...
```

Check `examples/output/` to see the extracted photos!

### Step 2: Process Your Own Scans

1. Scan your photos (save as JPG, PNG, or TIFF)
2. Run the splitter:

```bash
photo-splitter my_scan.jpg -o my_photos
```

3. The tool will:
   - Show you which photos it detected (press any key to continue)
   - For each photo, show a preview and ask if you want to save it:
     - Press `y` to save
     - Press `n` to skip
     - Press `q` to quit

### Step 3: Batch Processing

Process multiple scans at once:

```bash
# Put all your scans in a folder
mkdir my_scans
# ... copy your scan files here ...

# Process them all
photo-splitter my_scans/ -o all_photos --no-interactive
```

## Common Options

- `--no-interactive`: Skip previews, save all detected photos
- `--no-rotate`: Don't auto-correct rotation
- `--dust-removal`: Clean up dust, scratches, and film grain from old photos
- `--min-area 20000`: Only detect larger photos (adjust number as needed)

## Tips for Best Results

1. **High Resolution**: Scan at 300 DPI or higher
2. **Clean Background**: Use a clean, light-colored surface
3. **Good Spacing**: Leave space between photos
4. **Good Contrast**: Ensure photos stand out from the background
5. **Old Photos**: Use `--dust-removal` for vintage photos with dust and scratches

## Getting Help

```bash
photo-splitter --help
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [examples/README.md](examples/README.md) for more examples
- Run the tests: `python -m unittest discover tests`

Happy scanning! 📸
