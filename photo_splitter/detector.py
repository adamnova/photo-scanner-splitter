"""
Core photo detection and processing module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
import urllib.request


# Constants
ROTATION_THRESHOLD_DEGREES = 0.5  # Minimum rotation angle to apply correction


class PhotoDetector:
    """Detects individual photos in a scanned image"""
    
    def __init__(self, min_area: int = 10000, edge_threshold1: int = 50, edge_threshold2: int = 150,
                 dust_removal: bool = False):
                 face_confidence: float = 0.5):
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
        self.face_net = None
        self._load_face_detector()
    
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
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
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
        if abs(angle) < ROTATION_THRESHOLD_DEGREES:  # Don't rotate if angle is very small
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
        # Work with a copy to avoid modifying the original
        cleaned = image.copy()
        
        # Stage 1: Edge-preserving bilateral filtering for noise reduction
        # Preserves edges while smoothing flat regions - critical for quality
        cleaned = cv2.bilateralFilter(cleaned, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Stage 2: Apply non-local means denoising with higher quality settings
        # Stronger filtering for better quality (slower but much better results)
        cleaned = cv2.fastNlMeansDenoisingColored(
            cleaned, 
            None,
            h=15,           # Increased filter strength for better noise removal
            hColor=15,      # Increased color filter strength for better quality
            templateWindowSize=9,    # Larger template for better quality
            searchWindowSize=25      # Larger search area for superior results
        )
        
        # Stage 3: Advanced dust detection using multiple kernel sizes
        # Convert to grayscale for dust detection
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to grayscale for better dust detection
        gray_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Use multiple morphological operations with different kernel sizes
        # for more accurate dust detection
        dust_masks = []
        
        # Detect small dust (3x3 kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat_small = cv2.morphologyEx(gray_filtered, cv2.MORPH_TOPHAT, kernel_small)
        blackhat_small = cv2.morphologyEx(gray_filtered, cv2.MORPH_BLACKHAT, kernel_small)
        dust_masks.append(cv2.add(tophat_small, blackhat_small))
        
        # Detect medium dust (5x5 kernel)
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat_medium = cv2.morphologyEx(gray_filtered, cv2.MORPH_TOPHAT, kernel_medium)
        blackhat_medium = cv2.morphologyEx(gray_filtered, cv2.MORPH_BLACKHAT, kernel_medium)
        dust_masks.append(cv2.add(tophat_medium, blackhat_medium))
        
        # Detect larger dust/scratches (7x7 kernel)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat_large = cv2.morphologyEx(gray_filtered, cv2.MORPH_TOPHAT, kernel_large)
        blackhat_large = cv2.morphologyEx(gray_filtered, cv2.MORPH_BLACKHAT, kernel_large)
        dust_masks.append(cv2.add(tophat_large, blackhat_large))
        
        # Combine all dust masks for comprehensive detection
        dust_mask = cv2.add(cv2.add(dust_masks[0], dust_masks[1]), dust_masks[2])
        
        # Use adaptive thresholding for better mask quality
        dust_mask = cv2.adaptiveThreshold(
            dust_mask, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # Block size
            -2   # Constant subtracted from mean
        )
        
        # Morphological operations to refine the mask
        kernel_refine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Close small holes in dust regions
        dust_mask = cv2.morphologyEx(dust_mask, cv2.MORPH_CLOSE, kernel_refine, iterations=1)
        # Dilate slightly to ensure full dust coverage
        dust_mask = cv2.dilate(dust_mask, kernel_refine, iterations=1)
        
        # Stage 4: High-quality inpainting using Navier-Stokes algorithm
        # This method is slower but produces superior quality results
        if np.sum(dust_mask) > 0:  # Only inpaint if dust was detected
            # Use Navier-Stokes inpainting for better quality
            # Larger radius for better context and smoother results
            cleaned = cv2.inpaint(cleaned, dust_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        return cleaned
    def _load_face_detector(self):
        """
        Load the DNN-based face detection model
        Uses OpenCV's pre-trained ResNet SSD face detector
        """
        try:
            # Define model URLs and paths
            model_dir = os.path.join(os.path.expanduser("~"), ".photo_splitter")
            os.makedirs(model_dir, exist_ok=True)
            
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")
            model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            # Download prototxt if not exists
            if not os.path.exists(prototxt_path):
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            
            # Download model if not exists
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(model_url, model_path)
            
            # Load the model
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
        except Exception as e:
            # If loading fails, face detection will be disabled
            print(f"Warning: Could not load face detection model: {e}")
            self.face_net = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Detect faces in an image using deep learning
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing face information:
            - 'bbox': Tuple of (x, y, width, height)
            - 'confidence': Detection confidence score (0.0 to 1.0)
        """
        if self.face_net is None:
            return []
        
        h, w = image.shape[:2]
        
        # Prepare the image for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Pass the blob through the network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence threshold
            if confidence > self.face_confidence:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                width = endX - startX
                height = endY - startY
                
                # Only include valid detections
                if width > 0 and height > 0:
                    faces.append({
                        'bbox': (startX, startY, width, height),
                        'confidence': float(confidence)
                    })
        
        return faces
