"""Unified image loading and preprocessing pipeline for training and inference."""

from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import DEFAULT_IMAGE_SIZE, PROJECT_ROOT


def preprocess_image_array(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess a PIL Image or NumPy array into a normalized grayscale tensor.

    Processing contract:
    1. Standardize format to NumPy uint8.
    2. Convert RGB/RGBA/Multi-channel to single-channel Grayscale.
    3. Resize to target (width, height) using INTER_AREA interpolation.
    4. Normalize pixel intensities from [0, 255] to [0.0, 1.0] as float32.
    5. Ensure channel dimension is appended: (height, width, 1).

    Args:
        image: Input image as PIL Image or NumPy ndarray.
        target_size: Target (width, height) tuple for resizing. Defaults to (90, 90).

    Returns:
        Preprocessed NumPy ndarray of shape (height, width, 1) with dtype float32.

    Raises:
        ValueError: If input image is empty, has unsupported type, or unexpected dimensions.
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB first to resolve palette/LA modes reliably
        rgb_image = image.convert("RGB")
        np_image = np.array(rgb_image)
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image array is empty (size 0).")
        np_image = image.copy()
    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. Expected PIL.Image.Image or np.ndarray."
        )

    # Convert to single-channel 2D grayscale
    if np_image.ndim == 3:
        channels = np_image.shape[2]
        if channels == 4:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGBA2GRAY)
        elif channels == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        elif channels == 1:
            gray = np_image.squeeze(axis=-1)
        else:
            raise ValueError(f"Unsupported number of image channels: {channels}")
    elif np_image.ndim == 2:
        gray = np_image
    else:
        raise ValueError(f"Unexpected image dimensions: {np_image.ndim}D array.")

    # Resize to target (width, height)
    target_w, target_h = target_size
    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0] float32
    normalized = resized.astype(np.float32) / 255.0

    # Append channel dimension (height, width, 1)
    tensor = np.expand_dims(normalized, axis=-1)
    return tensor


def load_and_preprocess_image(
    file_path: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Read an image file from disk and apply preprocessing.

    Args:
        file_path: Path to the image file.
        target_size: Target (width, height) tuple. Defaults to (90, 90).

    Returns:
        Preprocessed NumPy ndarray of shape (height, width, 1) with dtype float32.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If image file is corrupt or unreadable.
    """
    path = Path(file_path)
    if not path.is_absolute() and not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path.resolve()}")

    try:
        with Image.open(path) as img:
            return preprocess_image_array(img, target_size=target_size)
    except Exception as exc:
        raise ValueError(f"Failed to load and process image at {path}: {exc}") from exc


def preprocess_for_inference(
    image: Union[Image.Image, np.ndarray, str, Path],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess image and add a batch dimension for single-sample model inference.

    Args:
        image: Image as path, PIL Image, or NumPy array.
        target_size: Target (width, height). Defaults to (90, 90).

    Returns:
        Preprocessed tensor of shape (1, height, width, 1) with dtype float32.
    """
    if isinstance(image, Path):
        tensor_hwc = load_and_preprocess_image(image, target_size=target_size)
    elif isinstance(image, str):
        # Check if string is a valid existing path; if not, raise ValueError
        path_candidate = Path(image)
        if path_candidate.exists() or (PROJECT_ROOT / path_candidate).exists():
            tensor_hwc = load_and_preprocess_image(path_candidate, target_size=target_size)
        else:
            raise ValueError(
                f"Unsupported image type or path not found: {image}. "
                "Expected PIL.Image.Image, np.ndarray, or a valid image file path."
            )
    elif isinstance(image, (Image.Image, np.ndarray)):
        tensor_hwc = preprocess_image_array(image, target_size=target_size)
    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. "
            "Expected PIL.Image.Image, np.ndarray, or Path."
        )

    # Add batch dimension: (1, height, width, 1)
    batched = np.expand_dims(tensor_hwc, axis=0)
    return batched
