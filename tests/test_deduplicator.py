"""
Tests for image deduplication module
"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from photo_splitter.deduplicator import ImageDeduplicator


class TestImageDeduplicator(unittest.TestCase):
    """Test cases for ImageDeduplicator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.deduplicator = ImageDeduplicator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_deduplicator_initialization(self):
        """Test deduplicator initializes with correct default values"""
        self.assertEqual(self.deduplicator.hash_size, 16)
        self.assertEqual(self.deduplicator.similarity_threshold, 0.95)

    def test_deduplicator_custom_initialization(self):
        """Test deduplicator initializes with custom values"""
        dedup = ImageDeduplicator(hash_size=8, similarity_threshold=0.90)
        self.assertEqual(dedup.hash_size, 8)
        self.assertEqual(dedup.similarity_threshold, 0.90)

    def test_compute_perceptual_hash_identical_images(self):
        """Test that identical images produce the same hash"""
        # Create a test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        hash1 = self.deduplicator.compute_perceptual_hash(image)
        hash2 = self.deduplicator.compute_perceptual_hash(image)

        self.assertEqual(hash1, hash2)

    def test_compute_perceptual_hash_different_images(self):
        """Test that different images produce different hashes"""
        image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        hash1 = self.deduplicator.compute_perceptual_hash(image1)
        hash2 = self.deduplicator.compute_perceptual_hash(image2)

        # With random images, hashes should be different with high probability
        # We can't guarantee this 100%, but it should be true most of the time
        self.assertNotEqual(hash1, hash2)

    def test_compute_perceptual_hash_grayscale(self):
        """Test that perceptual hash works with grayscale images"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        phash = self.deduplicator.compute_perceptual_hash(image)
        self.assertIsInstance(phash, str)
        self.assertGreater(len(phash), 0)

    def test_compute_quality_score_resolution(self):
        """Test that quality score increases with resolution"""
        small_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        large_image = np.ones((200, 200, 3), dtype=np.uint8) * 128

        score_small = self.deduplicator.compute_quality_score(small_image)
        score_large = self.deduplicator.compute_quality_score(large_image)

        self.assertGreater(score_large, score_small)

    def test_compute_quality_score_sharpness(self):
        """Test that quality score considers sharpness"""
        # Create a blurry image
        blurry = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Create a sharper image with edges
        sharp = np.ones((100, 100, 3), dtype=np.uint8) * 128
        sharp[40:60, :] = 255  # Add a horizontal edge

        score_blurry = self.deduplicator.compute_quality_score(blurry)
        score_sharp = self.deduplicator.compute_quality_score(sharp)

        # Sharp image should have higher score
        self.assertGreater(score_sharp, score_blurry)

    def test_compute_similarity_identical_hashes(self):
        """Test similarity computation for identical hashes"""
        hash1 = "abc123"
        hash2 = "abc123"

        similarity = self.deduplicator.compute_similarity(hash1, hash2)
        self.assertEqual(similarity, 1.0)

    def test_compute_similarity_different_hashes(self):
        """Test similarity computation for different hashes"""
        hash1 = "abc123"
        hash2 = "def456"

        similarity = self.deduplicator.compute_similarity(hash1, hash2)
        self.assertEqual(similarity, 0.0)

    def test_deduplicate_images_no_duplicates(self):
        """Test deduplication when there are no duplicates"""
        # Create distinct images
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image1[:, :50] = 255  # Half white

        image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2[:50, :] = 255  # Top white

        images = [("img1", image1), ("img2", image2)]

        unique, duplicates = self.deduplicator.deduplicate_images(images)

        self.assertEqual(len(unique), 2)
        self.assertEqual(len(duplicates), 0)

    def test_deduplicate_images_with_duplicates_keeps_higher_quality(self):
        """Test that deduplication keeps the higher quality version"""
        # Create two identical images but different sizes (one higher quality)
        small_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        large_image = cv2.resize(small_image, (200, 200))

        # Make them identical content-wise
        small_resized = cv2.resize(large_image, (100, 100))

        images = [("small", small_resized), ("large", large_image)]

        unique, duplicates = self.deduplicator.deduplicate_images(images)

        self.assertEqual(len(unique), 1)
        self.assertEqual(len(duplicates), 1)

        # The large image should be kept (higher quality)
        self.assertEqual(unique[0][0], "large")
        self.assertEqual(duplicates[0], "small")

    def test_deduplicate_images_empty_list(self):
        """Test deduplication with empty list"""
        unique, duplicates = self.deduplicator.deduplicate_images([])

        self.assertEqual(len(unique), 0)
        self.assertEqual(len(duplicates), 0)

    def test_deduplicate_images_single_image(self):
        """Test deduplication with single image"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        images = [("img1", image)]

        unique, duplicates = self.deduplicator.deduplicate_images(images)

        self.assertEqual(len(unique), 1)
        self.assertEqual(len(duplicates), 0)

    def test_deduplicate_image_paths_with_duplicates(self):
        """Test deduplication of image files"""
        # Create test images
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image2 = cv2.resize(image1, (200, 200))  # Same content, higher resolution

        path1 = os.path.join(self.temp_dir, "image1.jpg")
        path2 = os.path.join(self.temp_dir, "image2.jpg")

        cv2.imwrite(path1, image1)
        cv2.imwrite(path2, image2)

        from pathlib import Path

        paths = [Path(path1), Path(path2)]
        unique_paths, duplicate_paths = self.deduplicator.deduplicate_image_paths(paths)

        self.assertEqual(len(unique_paths), 1)
        self.assertEqual(len(duplicate_paths), 1)

        # Higher resolution image should be kept
        self.assertEqual(unique_paths[0].name, "image2.jpg")

    def test_deduplicate_image_paths_empty_list(self):
        """Test deduplication with empty path list"""
        from pathlib import Path

        unique, duplicates = self.deduplicator.deduplicate_image_paths([])

        self.assertEqual(len(unique), 0)
        self.assertEqual(len(duplicates), 0)

    def test_deduplicate_image_paths_invalid_images(self):
        """Test deduplication handles invalid image files"""
        # Create a text file (not an image)
        path1 = os.path.join(self.temp_dir, "not_an_image.txt")
        with open(path1, "w") as f:
            f.write("This is not an image")

        from pathlib import Path

        paths = [Path(path1)]
        unique, duplicates = self.deduplicator.deduplicate_image_paths(paths)

        # Should handle gracefully
        self.assertEqual(len(unique), 0)
        self.assertEqual(len(duplicates), 0)

    def test_deduplicate_images_multiple_duplicates(self):
        """Test deduplication with multiple duplicates of the same image"""
        # Create three identical images with different qualities
        image1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        image2 = cv2.resize(image1, (100, 100))
        image3 = cv2.resize(image1, (150, 150))

        # Make them have identical content
        image1_resized = cv2.resize(image3, (50, 50))
        image2_resized = cv2.resize(image3, (100, 100))

        images = [
            ("low", image1_resized),
            ("medium", image2_resized),
            ("high", image3),
        ]

        unique, duplicates = self.deduplicator.deduplicate_images(images)

        self.assertEqual(len(unique), 1)
        self.assertEqual(len(duplicates), 2)

        # Highest quality should be kept
        self.assertEqual(unique[0][0], "high")
        self.assertIn("low", duplicates)
        self.assertIn("medium", duplicates)


if __name__ == "__main__":
    unittest.main()
