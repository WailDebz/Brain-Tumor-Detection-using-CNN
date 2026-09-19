"""Image preprocessing pipeline for brain MRI inference."""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import DEFAULT_IMAGE_SIZE


def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess an input brain MRI image for CNN inference.

    Steps:
    1. Convert input to a NumPy array.
    2. Convert RGB/RGBA channels to grayscale if needed.
    3. Resize to target dimensions (width, height).
    4. Normalize pixel intensities to the [0.0, 1.0] range.
    5. Expand dimensions to (1, height, width, 1) to match model batch input.

    Args:
        image: Input image as PIL Image or NumPy ndarray.
        target_size: Target (width, height) tuple for resizing. Defaults to (90, 90).

    Returns:
        Preprocessed NumPy array of shape (1, height, width, 1) with dtype float32.

    Raises:
        ValueError: If image format is invalid or cannot be processed.
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB first to standardize format
        rgb_image = image.convert("RGB")
        np_image = np.array(rgb_image)
    elif isinstance(image, np.ndarray):
        np_image = image.copy()
    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. Expected PIL.Image.Image or np.ndarray."
        )

    if np_image.size == 0:
        raise ValueError("Provided image array is empty.")

    # Convert to grayscale based on number of channels
    if np_image.ndim == 3:
        if np_image.shape[2] == 4:  # RGBA
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2GRAY)
        elif np_image.shape[2] == 3:  # RGB
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        elif np_image.shape[2] == 1:
            gray_image = np_image.squeeze(axis=-1)
        else:
            raise ValueError(f"Unexpected channel count: {np_image.shape[2]}")
    elif np_image.ndim == 2:
        gray_image = np_image
    else:
        raise ValueError(f"Unexpected image dimensions: {np_image.ndim}D array.")

    # Resize image to target (width, height)
    target_w, target_h = target_size
    resized_image = cv2.resize(gray_image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0] float32
    normalized = resized_image.astype(np.float32) / 255.0

    # Expand dimensions to (1, height, width, 1)
    batched = np.expand_dims(normalized, axis=(0, -1))
    return batched
