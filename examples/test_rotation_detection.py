#!/usr/bin/env python3
"""
Example script demonstrating the new enhanced rotation detection feature
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import photo_splitter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from photo_splitter.detector import PhotoDetector


def create_test_image_with_text():
    """Create a test image with text-like horizontal patterns"""
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw horizontal bars to simulate text
    for y in range(80, 320, 40):
        cv2.rectangle(image, (50, y), (550, y+15), (0, 0, 0), -1)
    
    return image


def create_test_image_with_lines():
    """Create a test image with strong horizontal lines"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw horizontal lines
    for y in range(100, 300, 50):
        cv2.line(image, (50, y), (350, y), (0, 0, 0), 3)
    
    return image


def main():
    print("=" * 60)
    print("Enhanced Rotation Detection Feature Demo")
    print("=" * 60)
    
    detector = PhotoDetector()
    
    # Test 1: Image with text-like content
    print("\n[Test 1] Image with horizontal text-like patterns")
    print("-" * 60)
    image1 = create_test_image_with_text()
    
    # Detect rotation using enhanced method
    result = detector.detect_rotation_enhanced(image1)
    print(f"Enhanced detection result:")
    print(f"  Rotation angle: {result['angle']:.2f}°")
    print(f"  Confidence: {result['confidence']:.2f}")
    
    # Also test backward compatible method
    angle = detector.detect_rotation(image1)
    print(f"\nBackward compatible method:")
    print(f"  Rotation angle: {angle:.2f}°")
    
    # Test 2: Rotated image
    print("\n[Test 2] Same image rotated by 15 degrees")
    print("-" * 60)
    rotated_image = detector.rotate_image(image1, 15)
    result_rotated = detector.detect_rotation_enhanced(rotated_image)
    print(f"Enhanced detection result:")
    print(f"  Rotation angle: {result_rotated['angle']:.2f}° (should be ~-15°)")
    print(f"  Confidence: {result_rotated['confidence']:.2f}")
    
    # Test 3: Image with strong lines
    print("\n[Test 3] Image with horizontal lines")
    print("-" * 60)
    image2 = create_test_image_with_lines()
    result2 = detector.detect_rotation_enhanced(image2)
    print(f"Enhanced detection result:")
    print(f"  Rotation angle: {result2['angle']:.2f}°")
    print(f"  Confidence: {result2['confidence']:.2f}")
    
    # Test 4: Featureless image
    print("\n[Test 4] Uniform/featureless image")
    print("-" * 60)
    uniform_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
    result3 = detector.detect_rotation_enhanced(uniform_image)
    print(f"Enhanced detection result:")
    print(f"  Rotation angle: {result3['angle']:.2f}°")
    print(f"  Confidence: {result3['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("Key Features:")
    print("- Multiple detection strategies (Hough lines + Projection)")
    print("- Confidence scoring (0.0 to 1.0)")
    print("- Backward compatible with existing API")
    print("- Reliable detection even with complex content")
    print("=" * 60)


if __name__ == "__main__":
    main()
