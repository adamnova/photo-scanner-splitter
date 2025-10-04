"""
Tests for photo detector module
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from photo_splitter.detector import PhotoDetector


class TestPhotoDetector(unittest.TestCase):
    """Test cases for PhotoDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PhotoDetector()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_detector_initialization(self):
        """Test detector initializes with correct default values"""
        self.assertEqual(self.detector.min_area, 10000)
        self.assertEqual(self.detector.edge_threshold1, 50)
        self.assertEqual(self.detector.edge_threshold2, 150)
    
    def test_detector_custom_initialization(self):
        """Test detector initializes with custom values"""
        detector = PhotoDetector(min_area=5000, edge_threshold1=30, edge_threshold2=100)
        self.assertEqual(detector.min_area, 5000)
        self.assertEqual(detector.edge_threshold1, 30)
        self.assertEqual(detector.edge_threshold2, 100)
    
    def test_detect_photos_invalid_path(self):
        """Test detection with invalid image path raises error"""
        with self.assertRaises(ValueError):
            self.detector.detect_photos("/nonexistent/path.jpg")
    
    def test_detect_photos_simple_image(self):
        """Test detection on a simple synthetic image"""
        # Create a simple test image with a white rectangle (photo) on black background
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)
        
        # Save to temp file
        temp_path = os.path.join(self.temp_dir, "test_image.jpg")
        cv2.imwrite(temp_path, image)
        
        # Detect photos
        detected = self.detector.detect_photos(temp_path)
        
        # Should detect at least one photo
        self.assertGreater(len(detected), 0)
    
    def test_extract_photo(self):
        """Test photo extraction"""
        # Create test image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)
        
        temp_path = os.path.join(self.temp_dir, "test_extract.jpg")
        cv2.imwrite(temp_path, image)
        
        # Create a simple contour and bounding box
        bbox = (100, 100, 300, 300)
        contour = np.array([[100, 100], [400, 100], [400, 400], [100, 400]])
        
        # Extract photo
        extracted = self.detector.extract_photo(temp_path, contour, bbox)
        
        # Check dimensions
        self.assertEqual(extracted.shape[0], 300)  # height
        self.assertEqual(extracted.shape[1], 300)  # width
    
    def test_extract_photo_invalid_path(self):
        """Test that extract_photo raises error for invalid image path"""
        bbox = (100, 100, 300, 300)
        contour = np.array([[100, 100], [400, 100], [400, 400], [100, 400]])
        
        with self.assertRaises(ValueError):
            self.detector.extract_photo("/nonexistent/path.jpg", contour, bbox)
    
    def test_detect_rotation_no_rotation(self):
        """Test rotation detection on non-rotated image"""
        # Create a simple rectangular image
        image = np.ones((400, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (250, 350), (0, 0, 0), 2)
        
        angle = self.detector.detect_rotation(image)
        
        # Angle should be close to 0 for non-rotated image
        self.assertLess(abs(angle), 10)
    
    def test_rotate_image_no_rotation(self):
        """Test that images with angle < 0.5 are not rotated"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rotated = self.detector.rotate_image(image, 0.3)
        
        # Should return original image without rotation
        np.testing.assert_array_equal(image, rotated)
    
    def test_rotate_image_90_degrees(self):
        """Test 90 degree rotation"""
        # Create a non-square image to verify rotation
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        rotated = self.detector.rotate_image(image, 90)
        
        # After 90 degree rotation, dimensions should be swapped (approximately)
        # Width should become height and vice versa
        self.assertGreater(rotated.shape[0], rotated.shape[1])
    
    def test_rotate_image_small_angle(self):
        """Test small angle rotation"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        rotated = self.detector.rotate_image(image, 15)
        
        # Image should be rotated (dimensions may change slightly)
        self.assertIsNotNone(rotated)
        self.assertEqual(rotated.shape[2], 3)  # Should still be 3 channels


if __name__ == '__main__':
    unittest.main()
