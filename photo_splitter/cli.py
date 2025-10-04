"""
Command-line interface for photo splitter
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .detector import PhotoDetector


class PhotoSplitterCLI:
    """Interactive CLI for splitting and aligning photos"""
    
    def __init__(self, input_path: str, output_dir: str, auto_rotate: bool = True,
                 interactive: bool = True):
        """
        Initialize the CLI
        
        Args:
            input_path: Path to input image or directory
            output_dir: Directory to save output images
            auto_rotate: Whether to automatically detect and fix rotation
            interactive: Whether to show interactive validation
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.auto_rotate = auto_rotate
        self.interactive = interactive
        self.detector = PhotoDetector()
        
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
                    if abs(angle) > 0.5:
                        print(f"  Photo {idx}: Detected rotation of {angle:.1f}°")
                        photo = self.detector.rotate_image(photo, -angle)
                
                # Show preview and get user confirmation if interactive
                if self.interactive:
                    accepted = self._show_photo_preview(photo, idx, len(detected_photos))
                    if not accepted:
                        print(f"  Photo {idx}: Skipped by user")
                        continue
                
                # Save the photo
                output_path = self.output_dir / f"{base_name}_photo_{idx}.jpg"
                cv2.imwrite(str(output_path), photo)
                print(f"  Saved: {output_path.name}")
                saved_count += 1
                
            except Exception as e:
                print(f"  Error processing photo {idx}: {e}")
        
        return saved_count
    
    def _show_detection_preview(self, image_path: str, 
                               detected_photos: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]):
        """Show preview of detected photos with bounding boxes"""
        image = cv2.imread(image_path)
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
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = PhotoSplitterCLI(
        input_path=args.input,
        output_dir=args.output,
        auto_rotate=not args.no_rotate,
        interactive=not args.no_interactive
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
