"""
Command-line interface for photo splitter
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from .detector import PhotoDetector
from .location_identifier import LocationIdentifier


class PhotoSplitterCLI:
    """Interactive CLI for splitting and aligning photos"""
    
    def __init__(self, input_path: str, output_dir: str, auto_rotate: bool = True,
                 interactive: bool = True, identify_location: bool = False,
                 ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen2.5-vl:32b"):
        """
        Initialize the CLI
        
        Args:
            input_path: Path to input image or directory
            output_dir: Directory to save output images
            auto_rotate: Whether to automatically detect and fix rotation
            interactive: Whether to show interactive validation
            identify_location: Whether to identify photo locations using Ollama
            ollama_url: URL of the Ollama API server
            ollama_model: Name of the Ollama model to use
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.auto_rotate = auto_rotate
        self.interactive = interactive
        self.identify_location = identify_location
        self.detector = PhotoDetector()
        self.location_identifier = None
        
        # Initialize location identifier if requested
        if self.identify_location:
            try:
                self.location_identifier = LocationIdentifier(
                    ollama_url=ollama_url,
                    model=ollama_model
                )
                print(f"Location identification enabled using model: {ollama_model}")
            except Exception as e:
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
        except Exception as e:
            print(f"Error detecting photos: {e}")
            return 0
        
        if not detected_photos:
            print("  No photos detected in this image")
            return 0
        
        print(f"  Detected {len(detected_photos)} photo(s)")
        
        # Show preview if interactive
        if self.interactive:
            self._show_detection_preview(str(image_path), detected_photos)
        
        # Extract and save each photo
        saved_count = 0
        base_name = image_path.stem
        
        for idx, (contour, bbox) in enumerate(detected_photos, 1):
            try:
                # Extract photo
                photo = self.detector.extract_photo(str(image_path), contour, bbox)
                
                # Auto-rotate if enabled
                if self.auto_rotate:
                    angle = self.detector.detect_rotation(photo)
                    if abs(angle) > ROTATION_THRESHOLD_DEGREES:
                        print(f"  Photo {idx}: Detected rotation of {angle:.1f}°")
                        photo = self.detector.rotate_image(photo, -angle)
                
                # Identify location if enabled
                location_info = None
                if self.identify_location and self.location_identifier:
                    print(f"  Photo {idx}: Identifying location...")
                    try:
                        location_info = self.location_identifier.identify_location(photo)
                        if location_info.get('location'):
                            print(f"  Photo {idx}: Location: {location_info['location']} "
                                  f"(Confidence: {location_info.get('confidence', 'unknown')})")
                        else:
                            print(f"  Photo {idx}: Location could not be determined")
                        if location_info.get('description'):
                            print(f"  Photo {idx}: {location_info['description']}")
                    except Exception as e:
                        print(f"  Photo {idx}: Error identifying location: {e}")
                
                # Show preview and get user confirmation if interactive
                if self.interactive:
                    accepted = self._show_photo_preview(photo, idx, len(detected_photos))
                    if not accepted:
                        print(f"  Photo {idx}: Skipped by user")
                        continue
                
                # Save the photo
                output_path = self.output_dir / f"{base_name}_photo_{idx}.jpg"
                cv2.imwrite(str(output_path), photo)
                
                # Save location metadata if available
                if location_info and (location_info.get('location') or location_info.get('description')):
                    metadata_path = self.output_dir / f"{base_name}_photo_{idx}_location.txt"
                    with open(metadata_path, 'w') as f:
                        if location_info.get('location'):
                            f.write(f"Location: {location_info['location']}\n")
                        if location_info.get('confidence'):
                            f.write(f"Confidence: {location_info['confidence']}\n")
                        if location_info.get('description'):
                            f.write(f"Description: {location_info['description']}\n")
                    print(f"  Saved metadata: {metadata_path.name}")
                
                print(f"  Saved: {output_path.name}")
                saved_count += 1
                
            except Exception as e:
                print(f"  Error processing photo {idx}: {e}")
        
        return saved_count
    
    def _show_detection_preview(self, image_path: str, 
                               detected_photos: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]):
        """Show preview of detected photos with bounding boxes"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image from {image_path}")
            return
        
        preview = image.copy()
        
        for idx, (contour, bbox) in enumerate(detected_photos, 1):
            x, y, w, h = bbox
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(preview, str(idx), (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize for display if too large
        max_display_size = 1200
        h, w = preview.shape[:2]
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            preview = cv2.resize(preview, (new_w, new_h))
        
        cv2.imshow("Detected Photos (press any key to continue)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _show_photo_preview(self, photo: np.ndarray, photo_num: int, total: int) -> bool:
        """
        Show preview of extracted photo and get user confirmation
        
        Returns:
            True if user accepts the photo, False otherwise
        """
        # Resize for display if too large
        max_display_size = 800
        h, w = photo.shape[:2]
        display_photo = photo.copy()
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_photo = cv2.resize(display_photo, (new_w, new_h))
        
        window_name = f"Photo {photo_num}/{total} - Press 'y' to save, 'n' to skip, 'q' to quit"
        cv2.imshow(window_name, display_photo)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('y') or key == ord('Y'):
                return True
            elif key == ord('n') or key == ord('N'):
                return False
            elif key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                sys.exit(0)
            else:
                print("  Press 'y' to save, 'n' to skip, or 'q' to quit")
                cv2.imshow(window_name, display_photo)
    
    def run(self):
        """Run the photo splitter"""
        # Determine input files
        if self.input_path.is_file():
            image_files = [self.input_path]
        elif self.input_path.is_dir():
            # Find all image files in directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [f for f in self.input_path.iterdir() 
                          if f.suffix.lower() in image_extensions]
            image_files.sort()
        else:
            print(f"Error: {self.input_path} is not a valid file or directory")
            return
        
        if not image_files:
            print("No image files found")
            return
        
        print(f"Found {len(image_files)} image(s) to process")
        print(f"Output directory: {self.output_dir}")
        print(f"Auto-rotate: {'enabled' if self.auto_rotate else 'disabled'}")
        print(f"Interactive mode: {'enabled' if self.interactive else 'disabled'}")
        print(f"Location identification: {'enabled' if self.identify_location else 'disabled'}")
        
        # Process each image
        total_extracted = 0
        for image_file in image_files:
            count = self.process_image(image_file)
            total_extracted += count
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
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
  
  # Process with location identification using Ollama
  photo-splitter input.jpg -o output_photos --identify-location
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Input image file or directory containing scans')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output directory for extracted photos')
    parser.add_argument('--no-rotate', action='store_true',
                       help='Disable automatic rotation detection and correction')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive preview and confirmation')
    parser.add_argument('--min-area', type=int, default=10000,
                       help='Minimum area for photo detection (default: 10000)')
    parser.add_argument('--identify-location', action='store_true',
                       help='Enable location identification using Ollama LLM')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='URL of the Ollama API server (default: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='qwen2.5-vl:32b',
                       help='Ollama model to use for location identification (default: qwen2.5-vl:32b)')
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = PhotoSplitterCLI(
        input_path=args.input,
        output_dir=args.output,
        auto_rotate=not args.no_rotate,
        interactive=not args.no_interactive,
        identify_location=args.identify_location,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    # Update detector min_area if specified
    if args.min_area != 10000:
        cli.detector.min_area = args.min_area
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
