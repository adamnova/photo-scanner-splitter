"""
Tests for CLI module
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from photo_splitter.cli import PhotoSplitterCLI


class TestPhotoSplitterCLI(unittest.TestCase):
    """Test cases for PhotoSplitterCLI class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir)

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cli_initialization(self):
        """Test CLI initializes correctly"""
        cli = PhotoSplitterCLI(
            input_path=self.input_dir,
            output_dir=self.output_dir,
            auto_rotate=True,
            interactive=False,
        )

        self.assertEqual(cli.input_path, Path(self.input_dir))
        self.assertEqual(cli.output_dir, Path(self.output_dir))
        self.assertTrue(cli.auto_rotate)
        self.assertFalse(cli.interactive)

    def test_output_directory_creation(self):
        """Test that output directory is created"""
        output_path = os.path.join(self.temp_dir, "new_output")
        self.assertFalse(os.path.exists(output_path))

        PhotoSplitterCLI(input_path=self.input_dir, output_dir=output_path, interactive=False)

        self.assertTrue(os.path.exists(output_path))

    def test_process_image_no_photos(self):
        """Test processing image with no detectable photos"""
        # Create a blank image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        image_path = os.path.join(self.input_dir, "blank.jpg")
        cv2.imwrite(image_path, image)

        cli = PhotoSplitterCLI(input_path=image_path, output_dir=self.output_dir, interactive=False)

        count = cli.process_image(Path(image_path))
        self.assertEqual(count, 0)

    def test_process_image_with_photo(self):
        """Test processing image with a detectable photo"""
        # Create test image with a white rectangle (photo) on black background
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)

        image_path = os.path.join(self.input_dir, "test.jpg")
        cv2.imwrite(image_path, image)

        cli = PhotoSplitterCLI(input_path=image_path, output_dir=self.output_dir, interactive=False)

        count = cli.process_image(Path(image_path))

        # Should extract at least one photo
        self.assertGreaterEqual(count, 0)

        # Check output directory for saved files
        output_files = list(Path(self.output_dir).glob("*.jpg"))
        self.assertEqual(len(output_files), count)


if __name__ == "__main__":
    unittest.main()
