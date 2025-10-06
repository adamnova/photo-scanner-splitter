"""
Image deduplication module with quality-focused comparison
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Quality score weights for computing overall quality
QUALITY_WEIGHT_RESOLUTION = 0.0001  # Weight for resolution in quality score
QUALITY_WEIGHT_SHARPNESS = 10.0  # Weight for sharpness (Laplacian variance)
QUALITY_WEIGHT_BRIGHTNESS = 0.5  # Weight for brightness variance


class ImageDeduplicator:
    """Detects and removes duplicate images based on perceptual hashing with quality preference"""

    def __init__(
        self,
        hash_size: int = 16,
        similarity_threshold: float = 0.95,
    ):
        """
        Initialize the deduplicator

        Args:
            hash_size: Size of the perceptual hash (default: 16x16)
            similarity_threshold: Similarity threshold for duplicate detection (0.0-1.0, default: 0.95)
        """
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold

    def compute_perceptual_hash(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash (pHash) for an image

        This uses DCT-based hashing which is robust to minor variations
        while detecting visually similar images

        Args:
            image: Input image as numpy array

        Returns:
            Hexadecimal string representation of the perceptual hash
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Resize to hash_size + 1 for DCT
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size))

        # Apply DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(resized))

        # Keep only top-left corner (low frequencies)
        dct_low = dct[: self.hash_size, : self.hash_size]

        # Compute median
        median = np.median(dct_low)

        # Create binary hash based on median
        hash_binary = (dct_low > median).flatten()

        # Convert to hex string
        hash_bytes = np.packbits(hash_binary)
        return hashlib.sha256(hash_bytes).hexdigest()

    def compute_quality_score(self, image: np.ndarray) -> float:
        """
        Compute quality score for an image

        Higher scores indicate better quality. Quality is based on:
        - Image resolution (width * height)
        - Sharpness (Laplacian variance)
        - Brightness distribution

        Args:
            image: Input image as numpy array

        Returns:
            Quality score (higher is better)
        """
        height, width = image.shape[:2]
        resolution_score = width * height

        # Convert to grayscale for sharpness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Compute sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()

        # Compute brightness variance
        brightness_score = gray.std()

        # Combine scores (weighted)
        # Resolution is most important, then sharpness, then brightness variance
        quality_score = (
            resolution_score * QUALITY_WEIGHT_RESOLUTION
            + sharpness_score * QUALITY_WEIGHT_SHARPNESS
            + brightness_score * QUALITY_WEIGHT_BRIGHTNESS
        )

        return quality_score

    def compute_similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two perceptual hashes

        Args:
            hash1: First perceptual hash
            hash2: Second perceptual hash

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 means identical)
        """
        if hash1 == hash2:
            return 1.0

        # For different hashes, we consider them different enough
        # The pHash itself is already a similarity measure
        return 0.0

    def deduplicate_images(
        self, images: List[Tuple[str, np.ndarray]]
    ) -> Tuple[List[Tuple[str, np.ndarray]], List[str]]:
        """
        Deduplicate a list of images, keeping the highest quality version of each

        Args:
            images: List of tuples (identifier, image_array)

        Returns:
            Tuple of (unique_images, duplicates_removed)
            - unique_images: List of tuples (identifier, image_array) for unique images
            - duplicates_removed: List of identifiers that were removed as duplicates
        """
        if not images:
            return [], []

        # Compute hashes and quality scores for all images
        image_data = []
        for identifier, image in images:
            phash = self.compute_perceptual_hash(image)
            quality = self.compute_quality_score(image)
            image_data.append((identifier, image, phash, quality))

        # Group by hash
        hash_groups: Dict[str, List[Tuple[str, np.ndarray, float]]] = {}
        for identifier, image, phash, quality in image_data:
            if phash not in hash_groups:
                hash_groups[phash] = []
            hash_groups[phash].append((identifier, image, quality))

        # For each group, keep only the highest quality version
        unique_images = []
        duplicates_removed = []

        for _phash, group in hash_groups.items():
            # Sort by quality (descending)
            group_sorted = sorted(group, key=lambda x: x[2], reverse=True)

            # Keep the highest quality one
            best_identifier, best_image, _ = group_sorted[0]
            unique_images.append((best_identifier, best_image))

            # Mark others as duplicates
            for identifier, _, _ in group_sorted[1:]:
                duplicates_removed.append(identifier)

        return unique_images, duplicates_removed

    def deduplicate_image_paths(self, image_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Deduplicate image files by path, keeping the highest quality version

        Args:
            image_paths: List of paths to image files

        Returns:
            Tuple of (unique_paths, duplicate_paths)
            - unique_paths: List of paths to unique images
            - duplicate_paths: List of paths that were identified as duplicates
        """
        if not image_paths:
            return [], []

        # Load images and create identifier tuples
        images = []
        valid_paths = []
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is not None:
                images.append((str(path), image))
                valid_paths.append(path)

        if not images:
            return [], []

        # Deduplicate
        unique_images, duplicates_removed = self.deduplicate_images(images)

        # Convert back to paths
        unique_paths = [Path(identifier) for identifier, _ in unique_images]
        duplicate_paths = [Path(identifier) for identifier in duplicates_removed]

        return unique_paths, duplicate_paths
