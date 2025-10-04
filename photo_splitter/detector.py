"""
Core photo detection and processing module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class PhotoDetector:
    """Detects individual photos in a scanned image"""
    
    def __init__(self, min_area: int = 10000, edge_threshold1: int = 50, edge_threshold2: int = 150):
        """
        Initialize the photo detector
        
        Args:
            min_area: Minimum area for a photo to be considered valid (in pixels)
            edge_threshold1: First threshold for Canny edge detection
            edge_threshold2: Second threshold for Canny edge detection
        """
        self.min_area = min_area
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
    
    def detect_photos(self, image_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect individual photos in a scanned image
        
        Args:
            image_path: Path to the scanned image
            
        Returns:
            List of tuples (contour, bounding_box) for each detected photo
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
    
    def extract_photo(self, image_path: str, contour: np.ndarray, 
                     bounding_box: Tuple[int, int, int, int]) -> np.ndarray:
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
        x, y, w, h = bounding_box
        
        # Extract the region
        extracted = image[y:y+h, x:x+w]
        
        return extracted
    
    def detect_rotation(self, image: np.ndarray) -> float:
        """
        Detect the rotation angle of a photo using multiple strategies
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Rotation angle in degrees
        """
        # Use the enhanced rotation detection method
        result = self.detect_rotation_enhanced(image)
        return result['angle']
    
    def detect_rotation_enhanced(self, image: np.ndarray) -> dict:
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
        # Strategy 1: Hough line detection (existing method)
        hough_result = self._detect_rotation_hough(image)
        
        # Strategy 2: Projection profile analysis
        projection_result = self._detect_rotation_projection(image)
        
        # Combine results with weighting based on confidence
        angles = []
        weights = []
        
        if hough_result['confidence'] > 0.1:
            angles.append(hough_result['angle'])
            weights.append(hough_result['confidence'])
        
        if projection_result['confidence'] > 0.1:
            angles.append(projection_result['angle'])
            weights.append(projection_result['confidence'])
        
        if not angles:
            return {'angle': 0.0, 'confidence': 0.0}
        
        # Weighted average of angles
        total_weight = sum(weights)
        if total_weight == 0:
            return {'angle': 0.0, 'confidence': 0.0}
        
        weighted_angle = sum(a * w for a, w in zip(angles, weights)) / total_weight
        avg_confidence = total_weight / len(weights)
        
        return {
            'angle': float(weighted_angle),
            'confidence': float(min(1.0, avg_confidence))
        }
    
    def _detect_rotation_hough(self, image: np.ndarray) -> dict:
        """
        Detect rotation using Hough line transform
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with 'angle' and 'confidence'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) == 0:
            return {'angle': 0.0, 'confidence': 0.0}
        
        # Calculate dominant angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # Normalize angle to [-45, 45]
            while angle < -45:
                angle += 90
            while angle > 45:
                angle -= 90
            angles.append(angle)
        
        # Calculate median and confidence based on consistency
        median_angle = float(np.median(angles))
        
        # Confidence based on how concentrated the angles are
        std_dev = np.std(angles)
        confidence = max(0.0, 1.0 - (std_dev / 45.0))
        
        return {
            'angle': median_angle,
            'confidence': float(confidence)
        }
    
    def _detect_rotation_projection(self, image: np.ndarray) -> dict:
        """
        Detect rotation using projection profile analysis
        
        This method analyzes the variance in horizontal and vertical projections
        to detect text-like content and determine orientation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with 'angle' and 'confidence'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binarize the image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Test rotation angles
        test_angles = [0, 90, 180, 270]
        variances = []
        
        for angle in test_angles:
            # Rotate image
            if angle == 0:
                rotated = binary
            else:
                h, w = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(binary, M, (w, h))
            
            # Calculate horizontal projection (sum of each row)
            h_projection = np.sum(rotated, axis=1)
            # Calculate variance of projection
            variance = np.var(h_projection)
            variances.append(variance)
        
        # The correct orientation should have maximum variance in horizontal projection
        max_variance_idx = np.argmax(variances)
        detected_angle = test_angles[max_variance_idx]
        
        # Convert to rotation needed to correct
        if detected_angle == 0:
            correction_angle = 0.0
        elif detected_angle == 90:
            correction_angle = -90.0
        elif detected_angle == 180:
            correction_angle = 180.0
        else:  # 270
            correction_angle = 90.0
        
        # Normalize to [-45, 45] range
        while correction_angle > 45:
            correction_angle -= 90
        while correction_angle < -45:
            correction_angle += 90
        
        # Calculate confidence based on variance ratio
        max_var = variances[max_variance_idx]
        min_var = min(variances)
        if max_var > 0:
            confidence = min(1.0, (max_var - min_var) / max_var)
        else:
            confidence = 0.0
        
        return {
            'angle': float(correction_angle),
            'confidence': float(confidence)
        }
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an image by the given angle
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        if abs(angle) < 0.5:  # Don't rotate if angle is very small
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        return rotated
