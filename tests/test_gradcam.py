"""Unit tests for Gradient-weighted Class Activation Mapping (Grad-CAM)."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.explainability.gradcam import (
    GradCAMResult,
    compute_gradcam_heatmap,
    find_target_conv_layer,
    generate_gradcam_explanation,
    overlay_gradcam_heatmap,
)
from src.models.cnn import build_cnn_model
from src.models.transfer import build_transfer_model


@pytest.fixture
def synthetic_mri_image(tmp_path: Path) -> Path:
    """Create a temporary synthetic grayscale MRI image file."""
    img_array = np.full((120, 120, 3), fill_value=128, dtype=np.uint8)
    # Add a brighter region in center
    img_array[40:80, 40:80] = 220
    img = Image.fromarray(img_array)
    img_path = tmp_path / "synthetic_mri.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_mobilenetv2_model():
    """Build an un-trained MobileNetV2 model for deterministic testing."""
    return build_transfer_model(input_shape=(90, 90, 1), fine_tune=False)


@pytest.fixture
def mock_cnn_model():
    """Build an un-trained Custom CNN model for testing."""
    return build_cnn_model(input_shape=(90, 90, 1))


def test_find_target_conv_layer_mobilenetv2(mock_mobilenetv2_model):
    """Verify auto-detection and explicit target layer discovery for MobileNetV2."""
    layer_obj, layer_name = find_target_conv_layer(mock_mobilenetv2_model)
    assert layer_name in {"out_relu", "Conv_1"}
    assert layer_obj is not None

    # Test explicit layer
    explicit_obj, explicit_name = find_target_conv_layer(mock_mobilenetv2_model, "Conv_1")
    assert explicit_name == "Conv_1"


def test_find_target_conv_layer_custom_cnn(mock_cnn_model):
    """Verify target layer auto-detection on custom CNN baseline."""
    layer_obj, layer_name = find_target_conv_layer(mock_cnn_model)
    assert layer_name in {"relu3", "conv3_128"}
    assert layer_obj is not None


def test_find_target_conv_layer_invalid_name_raises_error(mock_mobilenetv2_model):
    """Verify specifying a non-existent layer name raises ValueError."""
    with pytest.raises(ValueError, match="not found in model"):
        find_target_conv_layer(mock_mobilenetv2_model, "non_existent_conv_layer_xyz")


def test_compute_gradcam_heatmap_mobilenetv2(mock_mobilenetv2_model):
    """Verify Grad-CAM heatmap computation on MobileNetV2 produces finite bounds in [0.0, 1.0]."""
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 90, 90, 1)).astype(np.float32)

    # Class 1 (Tumor)
    cam_raw, cam_norm, prob, layer = compute_gradcam_heatmap(
        model=mock_mobilenetv2_model,
        preprocessed_tensor=dummy_input,
        class_index=1,
    )

    assert cam_raw.ndim == 2
    assert cam_norm.shape == cam_raw.shape
    assert np.all(np.isfinite(cam_norm))
    assert 0.0 <= np.min(cam_norm)
    assert np.max(cam_norm) <= 1.0
    assert 0.0 <= prob <= 1.0
    assert layer in {"out_relu", "Conv_1"}


def test_compute_gradcam_heatmap_class_0_negative_explanation(mock_mobilenetv2_model):
    """Verify Grad-CAM computes gradients for Class 0 (No Tumor, targeting 1 - p)."""
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 90, 90, 1)).astype(np.float32)

    cam_raw_0, cam_norm_0, prob_0, _ = compute_gradcam_heatmap(
        model=mock_mobilenetv2_model,
        preprocessed_tensor=dummy_input,
        class_index=0,
    )

    assert np.all(np.isfinite(cam_norm_0))
    assert 0.0 <= np.min(cam_norm_0) <= np.max(cam_norm_0) <= 1.0


def test_compute_gradcam_heatmap_custom_cnn(mock_cnn_model):
    """Verify Grad-CAM computation on Custom CNN architecture."""
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 90, 90, 1)).astype(np.float32)

    cam_raw, cam_norm, prob, layer = compute_gradcam_heatmap(
        model=mock_cnn_model,
        preprocessed_tensor=dummy_input,
        class_index=1,
    )

    assert np.all(np.isfinite(cam_norm))
    assert 0.0 <= np.min(cam_norm) <= np.max(cam_norm) <= 1.0
    assert layer in {"relu3", "conv3_128"}


def test_overlay_gradcam_heatmap():
    """Verify overlay blending returns valid RGB arrays with correct dimensions."""
    orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap = np.full((10, 10), fill_value=0.5, dtype=np.float32)

    orig_rgb, resized_hm, overlay = overlay_gradcam_heatmap(orig_img, heatmap, alpha=0.4)

    assert orig_rgb.shape == (100, 100, 3)
    assert resized_hm.shape == (100, 100)
    assert overlay.shape == (100, 100, 3)
    assert orig_rgb.dtype == np.uint8
    assert overlay.dtype == np.uint8
    assert resized_hm.dtype == np.float32


def test_generate_gradcam_explanation_pipeline(mock_mobilenetv2_model, synthetic_mri_image):
    """Verify end-to-end generate_gradcam_explanation returns GradCAMResult."""
    result = generate_gradcam_explanation(
        model=mock_mobilenetv2_model,
        image=synthetic_mri_image,
        alpha=0.4,
    )

    assert isinstance(result, GradCAMResult)
    assert result.original_image.shape == (120, 120, 3)
    assert result.heatmap_resized.shape == (120, 120)
    assert result.overlay_image.shape == (120, 120, 3)
    assert 0.0 <= result.predicted_probability <= 1.0
    assert result.predicted_label in {"Tumor", "No Tumor"}
    assert result.target_class in {0, 1}


def test_gradcam_does_not_modify_model_weights(mock_mobilenetv2_model, synthetic_mri_image):
    """Verify Grad-CAM execution is strictly read-only and does not mutate weights."""
    initial_weights = [w.numpy().copy() for w in mock_mobilenetv2_model.weights]

    _ = generate_gradcam_explanation(mock_mobilenetv2_model, synthetic_mri_image)

    for w_init, w_after in zip(initial_weights, mock_mobilenetv2_model.weights):
        np.testing.assert_array_equal(w_init, w_after.numpy())


def test_gradcam_deterministic_execution(mock_mobilenetv2_model, synthetic_mri_image):
    """Verify Grad-CAM produces identical numerical heatmaps across repeated runs on same input."""
    res1 = generate_gradcam_explanation(mock_mobilenetv2_model, synthetic_mri_image, class_index=1)
    res2 = generate_gradcam_explanation(mock_mobilenetv2_model, synthetic_mri_image, class_index=1)

    np.testing.assert_allclose(res1.heatmap_resized, res2.heatmap_resized, atol=1e-6)
    assert res1.predicted_probability == res2.predicted_probability
