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
    
    def test_compute_image_hash(self):
        """Test computing image hash"""
        # Create a simple test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Compute hash
        hash_value = self.detector.compute_image_hash(image)
        
        # Hash should be a non-empty string
        self.assertIsNotNone(hash_value)
        self.assertIsInstance(hash_value, str)
        self.assertGreater(len(hash_value), 0)
    
    def test_compute_image_hash_consistency(self):
        """Test that same image produces same hash"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        
        hash1 = self.detector.compute_image_hash(image)
        hash2 = self.detector.compute_image_hash(image)
        
        # Same image should produce same hash
        self.assertEqual(hash1, hash2)
    
    def test_is_duplicate_same_image(self):
        """Test duplicate detection for identical images"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        # First image is not a duplicate
        self.assertFalse(self.detector.is_duplicate(image))
        
        # Mark as seen
        self.detector.mark_as_seen(image)
        
        # Same image should now be detected as duplicate
        self.assertTrue(self.detector.is_duplicate(image))
    
    def test_is_duplicate_different_images(self):
        """Test that different images are not marked as duplicates"""
        # Create two clearly different images with patterns
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        # Add distinct pattern to image1
        cv2.rectangle(image1, (10, 10), (30, 30), (255, 0, 0), -1)
        
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 200
        # Add different pattern to image2
        cv2.circle(image2, (50, 50), 20, (0, 255, 0), -1)
        
        # Mark first image as seen
        self.detector.mark_as_seen(image1)
        
        # Different image should not be a duplicate
        self.assertFalse(self.detector.is_duplicate(image2))
    
    def test_is_duplicate_similar_images(self):
        """Test duplicate detection for very similar images"""
        # Create an image with some content
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.rectangle(image1, (20, 20), (80, 80), (200, 150, 100), -1)
        cv2.circle(image1, (50, 50), 15, (100, 200, 150), -1)
        
        # Create a near-identical copy with very minor differences (JPEG compression artifacts)
        # Save and reload to simulate real-world scenario
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            cv2.imwrite(temp_path, image1)
            image2 = cv2.imread(temp_path)
            
            # Mark first image as seen
            self.detector.mark_as_seen(image1)
            
            # Near-identical image (after JPEG compression) should be detected as duplicate
            self.assertTrue(self.detector.is_duplicate(image2, hash_threshold=10))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_mark_as_seen(self):
        """Test marking images as seen"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        # Initially, seen_hashes should be empty
        self.assertEqual(len(self.detector.seen_hashes), 0)
        
        # Mark image as seen
        self.detector.mark_as_seen(image)
        
        # seen_hashes should now contain one hash
        self.assertEqual(len(self.detector.seen_hashes), 1)
    
    def test_reset_duplicate_tracking(self):
        """Test resetting duplicate tracking"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        
        # Mark image as seen
        self.detector.mark_as_seen(image)
        self.assertEqual(len(self.detector.seen_hashes), 1)
        
        # Reset tracking
        self.detector.reset_duplicate_tracking()
        
        # seen_hashes should be empty
        self.assertEqual(len(self.detector.seen_hashes), 0)
        
        # Image should no longer be detected as duplicate
        self.assertFalse(self.detector.is_duplicate(image))


if __name__ == '__main__':
    unittest.main()
