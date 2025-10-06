"""
Face detection module using deep learning
"""

import os
import urllib.request
from typing import Any, Dict, List

import cv2
import numpy as np


class FaceDetector:
    """Detects faces in images using deep learning"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the face detector

        Args:
            confidence_threshold: Minimum confidence threshold for face detection (0.0 to 1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.face_net = None
        self._load_face_detector()

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

        except (OSError, urllib.error.URLError, cv2.error) as e:
            # If loading fails, face detection will be disabled
            print(f"Warning: Could not load face detection model: {e}")
            self.face_net = None

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
        if self.face_net is None:
            return []

        h, w = image.shape[:2]

        # Prepare the image for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        # Pass the blob through the network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []

        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter by confidence threshold
            if confidence > self.confidence_threshold:
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
                    faces.append(
                        {"bbox": (startX, startY, width, height), "confidence": float(confidence)}
                    )

        return faces
