#!/usr/bin/env python3
"""
Example script demonstrating face detection functionality
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import photo_splitter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from photo_splitter.detector import PhotoDetector


def create_sample_image_with_faces():
    """Create a sample image with face-like patterns for testing"""
    # Create a blank image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 240
    
    # Draw multiple face-like patterns
    faces_positions = [
        (200, 200, 80, 100),  # x, y, w, h
        (500, 200, 80, 100),
        (350, 400, 80, 100),
    ]
    
    for (x, y, w, h) in faces_positions:
        # Draw face oval
        cv2.ellipse(image, (x, y), (w, h), 0, 0, 360, (200, 180, 150), -1)
        # Draw eyes
        cv2.circle(image, (x - 25, y - 20), 8, (50, 50, 50), -1)
        cv2.circle(image, (x + 25, y - 20), 8, (50, 50, 50), -1)
        # Draw nose
        cv2.line(image, (x, y - 10), (x, y + 20), (150, 130, 100), 2)
        # Draw mouth
        cv2.ellipse(image, (x, y + 30), (20, 10), 0, 0, 180, (100, 80, 80), 2)
    
    return image


def main():
    """Demonstrate face detection"""
    print("Face Detection Demo")
    print("=" * 50)
    
    # Create detector
    print("\n1. Initializing PhotoDetector with face detection...")
    detector = PhotoDetector(face_confidence=0.5)
    
    if detector.face_net is None:
        print("   Warning: Face detection model not loaded.")
        print("   This may be due to network issues or missing dependencies.")
        print("   Face detection will not be available.")
    else:
        print("   ✓ Face detection model loaded successfully!")
    
    # Create sample image
    print("\n2. Creating sample image with face-like patterns...")
    sample_image = create_sample_image_with_faces()
    
    # Detect faces
    print("\n3. Detecting faces in the sample image...")
    faces = detector.detect_faces(sample_image)
    
    print(f"   Found {len(faces)} face(s)")
    
    # Display results
    if len(faces) > 0:
        print("\n4. Face detection results:")
        for i, face in enumerate(faces, 1):
            bbox = face['bbox']
            confidence = face['confidence']
            print(f"   Face {i}:")
            print(f"     - Location: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
            print(f"     - Confidence: {confidence:.2%}")
            
            # Draw rectangle on the image
            x, y, w, h = bbox
            cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(sample_image, f"{confidence:.2%}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Save the result
        output_path = os.path.join(os.path.dirname(__file__), "face_detection_result.jpg")
        cv2.imwrite(output_path, sample_image)
        print(f"\n5. Result saved to: {output_path}")
        print("   Open the image to see detected faces marked with green boxes.")
    else:
        print("\n   No faces detected in the sample image.")
        if detector.face_net is None:
            print("   (This is expected since the model is not loaded)")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    
    # Additional example: Loading and detecting faces from a real image
    print("\n\nTo detect faces in your own images:")
    print("```python")
    print("from photo_splitter.detector import PhotoDetector")
    print("import cv2")
    print("")
    print("# Initialize detector")
    print("detector = PhotoDetector(face_confidence=0.5)")
    print("")
    print("# Load your image")
    print("image = cv2.imread('your_image.jpg')")
    print("")
    print("# Detect faces")
    print("faces = detector.detect_faces(image)")
    print("")
    print("# Process results")
    print("for face in faces:")
    print("    x, y, w, h = face['bbox']")
    print("    confidence = face['confidence']")
    print("    print(f'Face at ({x}, {y}) with confidence {confidence:.2%}')")
    print("```")


if __name__ == "__main__":
    main()
