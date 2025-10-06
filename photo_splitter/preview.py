"""
Preview and display utilities for the CLI
"""

import sys
from typing import List, Tuple

import cv2
import numpy as np

# Display size constants
MAX_PREVIEW_SIZE = 1200  # Maximum size for detection preview window
MAX_PHOTO_PREVIEW_SIZE = 800  # Maximum size for individual photo preview window


def show_detection_preview(
    image_path: str, detected_photos: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
):
    """
    Show preview of detected photos with bounding boxes

    Args:
        image_path: Path to the image file
        detected_photos: List of (contour, bbox) tuples

    Raises:
        ValueError: If image_path is empty or detected_photos is invalid
    """
    if not image_path:
        raise ValueError("Image path cannot be empty")
    if detected_photos is None:
        raise ValueError("Detected photos list cannot be None")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image from {image_path}")
        return

    preview = image.copy()

    for idx, (_contour, bbox) in enumerate(detected_photos, 1):
        x, y, w, h = bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            preview, str(idx), (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # Resize for display if too large
    h, w = preview.shape[:2]
    if max(h, w) > MAX_PREVIEW_SIZE:
        scale = MAX_PREVIEW_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        preview = cv2.resize(preview, (new_w, new_h))

    cv2.imshow("Detected Photos (press any key to continue)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_photo_preview(photo: np.ndarray, photo_num: int, total: int) -> bool:
    """
    Show preview of extracted photo and get user confirmation

    Args:
        photo: The extracted photo as numpy array
        photo_num: Current photo number
        total: Total number of photos

    Returns:
        True if user accepts the photo, False otherwise

    Raises:
        ValueError: If photo is invalid or numbers are out of range
    """
    if photo is None or photo.size == 0:
        raise ValueError("Photo cannot be None or empty")
    if len(photo.shape) < 2:
        raise ValueError("Photo must be at least 2-dimensional")
    if photo_num < 1 or total < 1:
        raise ValueError("Photo numbers must be positive")
    if photo_num > total:
        raise ValueError(f"Photo number {photo_num} cannot exceed total {total}")

    # Resize for display if too large
    h, w = photo.shape[:2]
    display_photo = photo.copy()
    if max(h, w) > MAX_PHOTO_PREVIEW_SIZE:
        scale = MAX_PHOTO_PREVIEW_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        display_photo = cv2.resize(display_photo, (new_w, new_h))

    window_name = f"Photo {photo_num}/{total} - Press 'y' to save, 'n' to skip, 'q' to quit"
    cv2.imshow(window_name, display_photo)

    while True:
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("y") or key == ord("Y"):
            return True
        elif key == ord("n") or key == ord("N"):
            return False
        elif key == ord("q") or key == ord("Q"):
            print("\nQuitting...")
            sys.exit(0)
        else:
            print("  Press 'y' to save, 'n' to skip, or 'q' to quit")
            cv2.imshow(window_name, display_photo)
