"""
Module for image processing and saving operations.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np


class ImageProcessor:
    """Handles image manipulation tasks such as adding noise and saving."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.noise_settings = config["noise_settings"]

    def add_background_noise(self, image: np.ndarray) -> np.ndarray:
        """Adds noise to the background (white pixels) of an image."""
        noisy_image = image.copy()
        background_mask = image == 255

        min_gray = self.noise_settings["min_gray_value"]
        max_gray = self.noise_settings["max_gray_value"]
        noise_intensity = self.noise_settings["noise_intensity"]

        random_noise_layer = np.random.randint(
            min_gray, max_gray + 1, size=image.shape, dtype=np.uint8
        )

        if noise_intensity > 0:
            texture_noise = self._generate_texture_noise(image.shape, noise_intensity)
            combined_noise_layer = (
                random_noise_layer * (1 - noise_intensity)
                + texture_noise * noise_intensity
            ).astype(np.uint8)
        else:
            combined_noise_layer = random_noise_layer

        noisy_image[background_mask] = combined_noise_layer[background_mask]

        return noisy_image

    def _generate_texture_noise(
        self, shape: Tuple[int, int], intensity: float
    ) -> np.ndarray:
        """Generates Perlin-like texture noise by combining multiple scales."""
        height, width = shape
        texture = np.zeros(shape, dtype=np.float32)

        scales = [50, 20, 5]  # Different scales for noise layers
        weights = [0.5, 0.3, 0.2]  # Weights for combining layers

        for scale, weight in zip(scales, weights):
            # Generate random noise and resize it to create a smoother effect
            random_field = np.random.randn(height // scale, width // scale)
            resized_noise = cv2.resize(
                random_field, (width, height), interpolation=cv2.INTER_CUBIC
            )
            texture += resized_noise * weight

        # Normalize texture to the desired gray value range
        min_gray = self.noise_settings["min_gray_value"]
        max_gray = self.noise_settings["max_gray_value"]

        # Prevent division by zero if texture is constant
        if texture.max() - texture.min() == 0:
            normalized_texture = np.full(
                shape, (min_gray + max_gray) / 2.0, dtype=np.float32
            )
        else:
            normalized_texture = (texture - texture.min()) / (
                texture.max() - texture.min()
            )
            normalized_texture = normalized_texture * (max_gray - min_gray) + min_gray

        return normalized_texture.astype(np.uint8)

    def save_image(self, image: np.ndarray, filepath: str) -> None:
        """Saves an image to the specified filepath."""
        cv2.imwrite(filepath, image)

    def apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Applies morphological operations to improve image quality (e.g., smoothing edges)."""
        kernel_size = 3
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.5)

        # Binary thresholding for sharp pore boundaries
        _, binary_image = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        return binary_image
