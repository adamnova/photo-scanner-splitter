"""
Core photo detection and processing module
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .face_detector import FaceDetector
from .image_processing import remove_dust, rotate_image
from .rotation_detector import RotationDetector


class PhotoDetector:
    """Detects individual photos in a scanned image"""

    def __init__(
        self,
        min_area: int = 10000,
        edge_threshold1: int = 50,
        edge_threshold2: int = 150,
        dust_removal: bool = False,
        face_confidence: float = 0.5,
    ):
        """
        Initialize the photo detector

        Args:
            min_area: Minimum area for a photo to be considered valid (in pixels)
            edge_threshold1: First threshold for Canny edge detection
            edge_threshold2: Second threshold for Canny edge detection
            dust_removal: Whether to apply dust removal to extracted photos
            face_confidence: Minimum confidence threshold for face detection (0.0 to 1.0)
        """
        self.min_area = min_area
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.dust_removal = dust_removal
        self.face_confidence = face_confidence

        # Initialize helper components
        self._rotation_detector = RotationDetector()
        self._face_detector = FaceDetector(confidence_threshold=face_confidence)

    def detect_photos(self, image_path: str) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect individual photos in a scanned image

        Args:
            image_path: Path to the scanned image

        Returns:
            List of tuples (contour, bounding_box) for each detected photo.
            Each bounding_box is a tuple of (x, y, w, h).
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)

        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and approximate to quadrilaterals
        detected_photos = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            detected_photos.append((approx, (x, y, w, h)))

        return detected_photos

    def extract_photo(
        self, image_path: str, contour: np.ndarray, bounding_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract a single photo from the scanned image

        Args:
            image_path: Path to the scanned image
            contour: Contour of the photo
            bounding_box: Bounding box (x, y, w, h)

        Returns:
            Extracted photo as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        x, y, w, h = bounding_box

        # Extract the region
        extracted = image[y : y + h, x : x + w]

        return extracted

    def detect_rotation(self, image: np.ndarray) -> float:
        """
        Detect the rotation angle of a photo

        Detect the rotation angle of a photo using multiple strategies

        Args:
            image: Input image as numpy array

        Returns:
            Rotation angle in degrees
        """
        return self._rotation_detector.detect_rotation(image)

    def detect_rotation_enhanced(self, image: np.ndarray) -> Dict[str, float]:
        """
        Enhanced rotation detection with confidence scoring

        Uses multiple strategies to reliably detect rotation:
        1. Hough line detection for strong edges
        2. Projection profile analysis for text-like content
        3. Variance-based detection

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary with 'angle' (float) and 'confidence' (float 0-1)
        """
        return self._rotation_detector.detect_rotation_enhanced(image)

    def _detect_rotation_hough(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect rotation using Hough line transform

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary with 'angle' and 'confidence'
        """
        return self._rotation_detector._detect_rotation_hough(image)

    def _detect_rotation_projection(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect rotation using projection profile analysis

        This method analyzes the variance in horizontal and vertical projections
        to detect text-like content and determine orientation.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary with 'angle' and 'confidence'
        """
        return self._rotation_detector._detect_rotation_projection(image)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an image by the given angle

        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        return rotate_image(image, angle)

    def remove_dust(self, image: np.ndarray) -> np.ndarray:
        """
        Remove dust and scratches from photos using state-of-the-art algorithms.

        This method uses a quality-focused multi-stage approach:
        1. Bilateral filtering for edge-preserving noise reduction
        2. Non-local means denoising for superior quality
        3. Advanced morphological operations for precise dust detection
        4. Navier-Stokes inpainting for highest quality restoration

        Args:
            image: Input image as numpy array

        Returns:
            Cleaned image with dust removed (quality-optimized)
        """
        return remove_dust(image)

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image using deep learning

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of dictionaries containing face information:
            - 'bbox': Tuple of (x, y, width, height)
            - 'confidence': Detection confidence score (0.0 to 1.0)
        """
        return self._face_detector.detect_faces(image)
