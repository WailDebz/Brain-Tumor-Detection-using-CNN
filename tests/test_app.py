"""Unit tests for Streamlit application validation and formatting helpers."""

from pathlib import Path
import io
import numpy as np
from PIL import Image
import pytest

from app.streamlit_app import (
    format_prediction_output,
    load_cached_model,
    validate_uploaded_image,
)
from src.config import PRIMARY_APP_MODEL_PATH


def test_primary_app_model_path_exists():
    """Verify primary application model path exists and points to MobileNetV2 frozen model."""
    assert PRIMARY_APP_MODEL_PATH.exists()
    assert PRIMARY_APP_MODEL_PATH.name == "brain_tumor_mobilenetv2_frozen.keras"


def test_load_cached_model_missing_file():
    """Verify load_cached_model handles missing file gracefully without crashing."""
    result = load_cached_model(Path("models/non_existent_app_model.keras"))
    assert result is None


def test_format_prediction_output_tumor():
    """Verify format_prediction_output formats probabilities >= threshold as Tumor."""
    label, pct = format_prediction_output(0.824, threshold=0.5)
    assert label == "Tumor"
    assert pct == pytest.approx(82.4, abs=0.01)


def test_format_prediction_output_no_tumor():
    """Verify format_prediction_output formats probabilities < threshold as No Tumor."""
    label, pct = format_prediction_output(0.153, threshold=0.5)
    assert label == "No Tumor"
    assert pct == pytest.approx(15.3, abs=0.01)


def test_validate_uploaded_image_none():
    """Verify validate_uploaded_image handles None input cleanly."""
    img, err = validate_uploaded_image(None)
    assert img is None
    assert "No file provided" in err


def test_validate_uploaded_image_valid_rgb():
    """Verify validate_uploaded_image decodes valid RGB in-memory file."""
    buf = io.BytesIO()
    raw_img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    raw_img.save(buf, format="PNG")
    buf.seek(0)

    img, err = validate_uploaded_image(buf)
    assert err is None
    assert img is not None
    assert img.size == (100, 100)


def test_validate_uploaded_image_valid_rgba():
    """Verify validate_uploaded_image decodes valid RGBA in-memory file."""
    buf = io.BytesIO()
    raw_img = Image.new("RGBA", (80, 80), color=(50, 100, 150, 200))
    raw_img.save(buf, format="PNG")
    buf.seek(0)

    img, err = validate_uploaded_image(buf)
    assert err is None
    assert img is not None


def test_validate_uploaded_image_corrupt_bytes():
    """Verify validate_uploaded_image returns friendly error for corrupt bytes."""
    buf = io.BytesIO(b"not a valid image content string")
    img, err = validate_uploaded_image(buf)
    assert img is None
    assert "Could not decode" in err
