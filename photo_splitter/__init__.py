"""
Photo Scanner Splitter - A tool to split and align scanned photos
"""

__version__ = "0.1.0"

# Import main classes for easier access
from .detector import PhotoDetector
from .face_detector import FaceDetector
from .image_processing import ROTATION_THRESHOLD_DEGREES, remove_dust, rotate_image
from .rotation_detector import RotationDetector

__all__ = [
    "PhotoDetector",
    "FaceDetector",
    "RotationDetector",
    "rotate_image",
    "remove_dust",
    "ROTATION_THRESHOLD_DEGREES",
]
