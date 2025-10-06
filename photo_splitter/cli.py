"""
Command-line interface for photo splitter
"""

import argparse
import sys
from pathlib import Path

from .deduplicator import ImageDeduplicator
from .detector import PhotoDetector
from .location_identifier import LocationIdentifier
from .preview import show_detection_preview, show_photo_preview
from .workflow import (
    deduplicate_extracted_photos,
    identify_photo_location,
    process_single_photo,
    save_photo_with_metadata,
)


class PhotoSplitterCLI:
    """Interactive CLI for splitting and aligning photos"""

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        auto_rotate: bool = True,
        interactive: bool = True,
        dust_removal: bool = False,
        identify_location: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5-vl:32b",
        deduplicate_source: bool = False,
        deduplicate_photos: bool = False,
    ):
        """
        Initialize the CLI

        Args:
            input_path: Path to input image or directory
            output_dir: Directory to save output images
            auto_rotate: Whether to automatically detect and fix rotation
            interactive: Whether to show interactive validation
            dust_removal: Whether to apply dust removal to extracted photos
            identify_location: Whether to identify photo locations using Ollama
            ollama_url: URL of the Ollama API server
            ollama_model: Name of the Ollama model to use
            deduplicate_source: Whether to deduplicate source images before processing
            deduplicate_photos: Whether to deduplicate extracted photos
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.auto_rotate = auto_rotate
        self.interactive = interactive
        self.dust_removal = dust_removal
        self.identify_location = identify_location
        self.deduplicate_source = deduplicate_source
        self.deduplicate_photos = deduplicate_photos
        self.detector = PhotoDetector(dust_removal=dust_removal)
        self.location_identifier = None
        self.deduplicator = (
            ImageDeduplicator() if (deduplicate_source or deduplicate_photos) else None
        )

        # Initialize location identifier if requested
        if self.identify_location:
            try:
                self.location_identifier = LocationIdentifier(
                    ollama_url=ollama_url, model=ollama_model
                )
                print(f"Location identification enabled using model: {ollama_model}")
            except (ConnectionError, OSError, ValueError) as e:
                print(f"Warning: Could not initialize location identifier: {e}")
                print("Continuing without location identification...")
                self.identify_location = False

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path: Path) -> int:
        """
        Process a single scanned image

        Args:
            image_path: Path to the image file

        Returns:
            Number of photos extracted
        """
        print(f"\nProcessing: {image_path.name}")

        # Detect photos
        try:
            detected_photos = self.detector.detect_photos(str(image_path))
        except (ValueError, OSError) as e:
            print(f"Error detecting photos: {e}")
            return 0

        if not detected_photos:
            print("  No photos detected in this image")
            return 0

        print(f"  Detected {len(detected_photos)} photo(s)")

        # Show preview if interactive
        if self.interactive:
            show_detection_preview(str(image_path), detected_photos)

        # Extract all photos first
        extracted_photos = []
        base_name = image_path.stem

        for idx, (contour, bbox) in enumerate(detected_photos, 1):
            photo = process_single_photo(
                self.detector,
                str(image_path),
                contour,
                bbox,
                self.auto_rotate,
                self.dust_removal,
            )
            if photo is not None:
                extracted_photos.append((idx, photo))
            else:
                print(f"  Error processing photo {idx}")

        # Deduplicate extracted photos if enabled
        if self.deduplicate_photos:
            extracted_photos = deduplicate_extracted_photos(
                self.deduplicator, extracted_photos, base_name
            )

        # Process and save each unique photo
        saved_count = 0
        for idx, photo in extracted_photos:
            try:
                # Identify location if enabled
                location_info = None
                if self.identify_location:
                    location_info = identify_photo_location(self.location_identifier, photo, idx)

                # Show preview and get user confirmation if interactive
                if self.interactive:
                    accepted = show_photo_preview(photo, idx, len(extracted_photos))
                    if not accepted:
                        print(f"  Photo {idx}: Skipped by user")
                        continue

                # Save the photo
                output_path = self.output_dir / f"{base_name}_photo_{idx}.jpg"
                if save_photo_with_metadata(photo, output_path, location_info):
                    saved_count += 1

            except (ValueError, OSError) as e:
                print(f"  Error processing photo {idx}: {e}")

        return saved_count

    def run(self):
        """Run the photo splitter"""
        # Validate input path exists
        if not self.input_path.exists():
            print(f"Error: {self.input_path} does not exist")
            return

        # Determine input files
        if self.input_path.is_file():
            image_files = [self.input_path]
        elif self.input_path.is_dir():
            # Find all image files in directory
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            image_files = [
                f for f in self.input_path.iterdir() if f.suffix.lower() in image_extensions
            ]
            image_files.sort()
        else:
            print(f"Error: {self.input_path} is not a valid file or directory")
            return

        if not image_files:
            print("No image files found")
            return

        # Deduplicate source images if enabled
        if self.deduplicate_source and self.deduplicator and len(image_files) > 1:
            print(f"\nDeduplicating {len(image_files)} source image(s)...")
            unique_files, duplicate_files = self.deduplicator.deduplicate_image_paths(image_files)

            if duplicate_files:
                print(
                    f"Found {len(duplicate_files)} duplicate source image(s) (keeping higher quality versions):"
                )
                for dup in duplicate_files:
                    print(f"  - {dup.name}")
                image_files = unique_files
            else:
                print("No duplicate source images found")

        print(f"\nFound {len(image_files)} image(s) to process")
        print(f"Output directory: {self.output_dir}")
        print(f"Auto-rotate: {'enabled' if self.auto_rotate else 'disabled'}")
        print(f"Dust removal: {'enabled' if self.dust_removal else 'disabled'}")
        print(f"Interactive mode: {'enabled' if self.interactive else 'disabled'}")
        print(f"Source deduplication: {'enabled' if self.deduplicate_source else 'disabled'}")
        print(f"Photo deduplication: {'enabled' if self.deduplicate_photos else 'disabled'}")
        print(f"Location identification: {'enabled' if self.identify_location else 'disabled'}")

        # Process each image
        total_extracted = 0
        for image_file in image_files:
            count = self.process_image(image_file)
            total_extracted += count

        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Total photos extracted: {total_extracted}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="Split and align photos from scanned images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single scan with interactive mode
  photo-splitter input.jpg -o output_photos

  # Process all images in a directory without interaction
  photo-splitter scans/ -o output_photos --no-interactive

  # Process without auto-rotation
  photo-splitter input.jpg -o output_photos --no-rotate

  # Process with dust removal enabled
  photo-splitter input.jpg -o output_photos --dust-removal

  # Process with location identification using Ollama
  photo-splitter input.jpg -o output_photos --identify-location

  # Process with source image deduplication
  photo-splitter scans/ -o output_photos --deduplicate-source

  # Process with photo deduplication (after extraction)
  photo-splitter input.jpg -o output_photos --deduplicate-photos
        """,
    )

    parser.add_argument("input", type=str, help="Input image file or directory containing scans")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output directory for extracted photos"
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable automatic rotation detection and correction",
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="Disable interactive preview and confirmation"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10000,
        help="Minimum area for photo detection (default: 10000)",
    )
    parser.add_argument(
        "--dust-removal", action="store_true", help="Enable dust and scratch removal from photos"
    )
    parser.add_argument(
        "--identify-location",
        action="store_true",
        help="Enable location identification using Ollama LLM",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="URL of the Ollama API server (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen2.5-vl:32b",
        help="Ollama model to use for location identification (default: qwen2.5-vl:32b)",
    )
    parser.add_argument(
        "--deduplicate-source",
        action="store_true",
        help="Enable deduplication of source images before processing (keeps highest quality)",
    )
    parser.add_argument(
        "--deduplicate-photos",
        action="store_true",
        help="Enable deduplication of extracted photos (keeps highest quality)",
    )

    args = parser.parse_args()

    # Create and run CLI
    cli = PhotoSplitterCLI(
        input_path=args.input,
        output_dir=args.output,
        auto_rotate=not args.no_rotate,
        interactive=not args.no_interactive,
        dust_removal=args.dust_removal,
        identify_location=args.identify_location,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        deduplicate_source=args.deduplicate_source,
        deduplicate_photos=args.deduplicate_photos,
    )

    # Update detector min_area if specified
    if args.min_area != 10000:
        cli.detector.min_area = args.min_area

    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
