"""
Image processing workflow for the CLI
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import PhotoDetector
from .image_processing import ROTATION_THRESHOLD_DEGREES
from .location_identifier import LocationIdentifier


def process_single_photo(
    detector: PhotoDetector,
    image_path: str,
    contour: np.ndarray,
    bbox: Tuple[int, int, int, int],
    auto_rotate: bool,
    dust_removal: bool,
) -> Optional[np.ndarray]:
    """
    Process a single photo extraction

    Args:
        detector: PhotoDetector instance
        image_path: Path to source image
        contour: Photo contour
        bbox: Bounding box (x, y, w, h)
        auto_rotate: Whether to apply rotation correction
        dust_removal: Whether to apply dust removal

    Returns:
        Processed photo as numpy array, or None if processing failed
    """
    try:
        # Extract photo
        photo = detector.extract_photo(image_path, contour, bbox)

        # Apply dust removal if enabled
        if dust_removal:
            photo = detector.remove_dust(photo)

        # Auto-rotate if enabled
        if auto_rotate:
            angle = detector.detect_rotation(photo)
            if abs(angle) > ROTATION_THRESHOLD_DEGREES:
                photo = detector.rotate_image(photo, -angle)

        return photo

    except (ValueError, OSError, cv2.error) as e:
        print(f"  Error during photo processing: {e}")
        return None


def identify_photo_location(
    location_identifier: Optional[LocationIdentifier], photo: np.ndarray, photo_idx: int
) -> Optional[dict]:
    """
    Identify the location of a photo using Ollama

    Args:
        location_identifier: LocationIdentifier instance or None
        photo: Photo as numpy array
        photo_idx: Photo index for logging

    Returns:
        Location info dictionary or None
    """
    if location_identifier is None:
        return None

    print(f"  Photo {photo_idx}: Identifying location...")
    try:
        location_info = location_identifier.identify_location(photo)
        if location_info.get("location"):
            print(
                f"  Photo {photo_idx}: Location: {location_info['location']} "
                f"(Confidence: {location_info.get('confidence', 'unknown')})"
            )
        else:
            print(f"  Photo {photo_idx}: Location could not be determined")
        if location_info.get("description"):
            print(f"  Photo {photo_idx}: {location_info['description']}")
        return location_info
    except (ConnectionError, ValueError, RuntimeError) as e:
        print(f"  Photo {photo_idx}: Error identifying location: {e}")
        return None


def save_photo_with_metadata(
    photo: np.ndarray,
    output_path: Path,
    location_info: Optional[dict] = None,
) -> bool:
    """
    Save photo and optional location metadata

    Args:
        photo: Photo as numpy array
        output_path: Path to save the photo
        location_info: Optional location information

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Save the photo
        cv2.imwrite(str(output_path), photo)
        print(f"  Saved: {output_path.name}")

        # Save location metadata if available
        if location_info and (location_info.get("location") or location_info.get("description")):
            metadata_path = output_path.parent / f"{output_path.stem}_location.txt"
            with open(metadata_path, "w") as f:
                if location_info.get("location"):
                    f.write(f"Location: {location_info['location']}\n")
                if location_info.get("confidence"):
                    f.write(f"Confidence: {location_info['confidence']}\n")
                if location_info.get("description"):
                    f.write(f"Description: {location_info['description']}\n")
            print(f"  Saved metadata: {metadata_path.name}")

        return True

    except (OSError, cv2.error) as e:
        print(f"  Error saving photo: {e}")
        return False


def deduplicate_extracted_photos(
    deduplicator, extracted_photos: List[Tuple[int, np.ndarray]], base_name: str
) -> List[Tuple[int, np.ndarray]]:
    """
    Deduplicate extracted photos

    Args:
        deduplicator: ImageDeduplicator instance
        extracted_photos: List of (idx, photo) tuples
        base_name: Base name for photo identifiers

    Returns:
        Deduplicated list of (idx, photo) tuples
    """
    if not deduplicator or not extracted_photos:
        return extracted_photos

    print(f"  Deduplicating {len(extracted_photos)} extracted photo(s)...")
    photos_to_process = [(f"{base_name}_photo_{idx}", photo) for idx, photo in extracted_photos]
    unique_photos, duplicates = deduplicator.deduplicate_images(photos_to_process)

    if duplicates:
        print(f"  Removed {len(duplicates)} duplicate photo(s) (kept higher quality versions)")
        for dup in duplicates:
            print(f"    - {dup}")

    # Parse back the idx from identifier
    result = []
    for identifier, photo in unique_photos:
        # Extract idx from identifier like "scan_photo_1"
        idx_str = identifier.split("_photo_")[-1]
        idx = int(idx_str)
        result.append((idx, photo))

    return result
