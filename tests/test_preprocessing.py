"""Unit tests for the image preprocessing pipeline."""

import numpy as np
import pytest
from PIL import Image

from src.inference.preprocessing import preprocess_image


def test_preprocess_rgb_pil_image():
    """Verify preprocessing on standard RGB PIL image."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    processed = preprocess_image(img, target_size=(90, 90))

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 90, 90, 1)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min() <= processed.max() <= 1.0


def test_preprocess_rgba_pil_image():
    """Verify preprocessing on RGBA PIL image with alpha channel."""
    img = Image.new("RGBA", (150, 150), color=(200, 100, 50, 255))
    processed = preprocess_image(img, target_size=(90, 90))

    assert processed.shape == (1, 90, 90, 1)
    assert processed.dtype == np.float32


def test_preprocess_grayscale_numpy_array():
    """Verify preprocessing on 2D grayscale NumPy array."""
    raw_array = np.random.randint(0, 256, size=(120, 120), dtype=np.uint8)
    processed = preprocess_image(raw_array, target_size=(90, 90))

    assert processed.shape == (1, 90, 90, 1)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min() <= processed.max() <= 1.0


def test_preprocess_rgb_numpy_array():
    """Verify preprocessing on 3D RGB NumPy array."""
    raw_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    processed = preprocess_image(raw_array, target_size=(90, 90))

    assert processed.shape == (1, 90, 90, 1)
    assert processed.dtype == np.float32


def test_preprocess_invalid_type_raises_value_error():
    """Verify passing an invalid type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported image type"):
        preprocess_image("invalid_string_input")  # type: ignore


def test_preprocess_empty_array_raises_value_error():
    """Verify passing an empty numpy array raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        preprocess_image(np.array([]))


def test_preprocess_invalid_ndim_raises_value_error():
    """Verify passing 1D array raises ValueError."""
    with pytest.raises(ValueError, match="Unexpected image dimensions"):
        preprocess_image(np.zeros((10,)))
