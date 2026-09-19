"""Image preprocessing pipeline for brain MRI inference."""

from typing import Tuple, Union
import numpy as np
from PIL import Image

from src.config import DEFAULT_IMAGE_SIZE
from src.data.preprocessing import preprocess_for_inference, preprocess_image_array


def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess an input brain MRI image for CNN inference.

    Delegates to unified src.data.preprocessing pipeline.

    Args:
        image: Input image as PIL Image or NumPy ndarray.
        target_size: Target (width, height) tuple for resizing. Defaults to (90, 90).

    Returns:
        Preprocessed NumPy array of shape (1, height, width, 1) with dtype float32.
    """
    return preprocess_for_inference(image, target_size=target_size)

