"""
Core photo detection and processing module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class PhotoDetector:
    """Detects individual photos in a scanned image"""
    
    def __init__(self, min_area: int = 10000, edge_threshold1: int = 50, edge_threshold2: int = 150,
                 dust_removal: bool = False):
        """
        Initialize the photo detector
        
        Args:
            min_area: Minimum area for a photo to be considered valid (in pixels)
            edge_threshold1: First threshold for Canny edge detection
            edge_threshold2: Second threshold for Canny edge detection
            dust_removal: Whether to apply dust removal to extracted photos
        """
        self.min_area = min_area
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.dust_removal = dust_removal
    
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
        Detect the rotation angle of a photo
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Rotation angle in degrees
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
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
        
        # Return median angle
        return float(np.median(angles))
    
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
    
    def remove_dust(self, image: np.ndarray) -> np.ndarray:
        """
        Remove dust and scratches from photos using state-of-the-art algorithms.
        
        This method uses a multi-stage approach:
        1. Non-local means denoising for general noise reduction
        2. Morphological operations to detect dust spots
        3. Inpainting to remove detected dust and scratches
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Cleaned image with dust removed
        """
        # Work with a copy to avoid modifying the original
        cleaned = image.copy()
        
        # Stage 1: Apply non-local means denoising for general noise reduction
        # This is effective for subtle dust and film grain
        cleaned = cv2.fastNlMeansDenoisingColored(
            cleaned, 
            None,
            h=10,           # Filter strength for luminance
            hColor=10,      # Filter strength for color
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Stage 2: Detect dust spots using morphological operations
        # Convert to grayscale for dust detection
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to identify small dark or bright spots (dust)
        # Use top-hat transform to find bright spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Use black-hat transform to find dark spots
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both to get all dust spots
        dust_mask = cv2.add(tophat, blackhat)
        
        # Threshold to create binary mask of dust
        _, dust_mask = cv2.threshold(dust_mask, 10, 255, cv2.THRESH_BINARY)
        
        # Dilate the mask slightly to ensure we capture the full extent of dust spots
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dust_mask = cv2.dilate(dust_mask, kernel_dilate, iterations=1)
        
        # Stage 3: Inpainting to remove detected dust
        # Use Telea inpainting algorithm (fast and effective for small defects)
        if np.sum(dust_mask) > 0:  # Only inpaint if dust was detected
            cleaned = cv2.inpaint(cleaned, dust_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        return cleaned
