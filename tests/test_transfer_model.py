"""Unit tests for the transfer learning model (MobileNetV2 feature extractor)."""

from pathlib import Path
import numpy as np
import pytest

from src.models.transfer import build_transfer_model


def test_build_transfer_model_structure():
    """Verify transfer model layer hierarchy and frozen backbone configuration."""
    import tensorflow as tf

    model = build_transfer_model(input_shape=(90, 90, 1), fine_tune=False)

    assert isinstance(model, tf.keras.Model)
    assert model.name == "brain_tumor_transfer_mobilenetv2"
    assert model.input_shape == (None, 90, 90, 1)
    assert model.output_shape == (None, 1)

    layer_names = [layer.name for layer in model.layers]
    assert "input_mri_slice" in layer_names
    assert "grayscale_to_rgb" in layer_names
    assert "rescale_to_mobilenet_range" in layer_names
    assert "mobilenetv2_1.00_224" in layer_names
    assert "global_avg_pool" in layer_names
    assert "dense_features" in layer_names
    assert "output_probability" in layer_names


def test_transfer_model_frozen_backbone_parameter_counts():
    """Verify backbone weights are non-trainable while classification head is trainable."""
    model = build_transfer_model(input_shape=(90, 90, 1), fine_tune=False)

    total_params = model.count_params()
    trainable_params = sum(v.numpy().size for v in model.trainable_variables)
    non_trainable_params = sum(v.numpy().size for v in model.non_trainable_variables)

    # MobileNetV2 backbone has 2,257,984 non-trainable weights (+2 Rescaling parameters = 2,257,986)
    # Head has Dense(64) [1280*64+64 = 81,984] + Dense(1) [64*1+1 = 65] = 82,049 trainable params
    assert total_params == 2_340_033
    assert trainable_params == 82_049
    assert non_trainable_params == 2_257_986


def test_transfer_model_forward_pass_and_bounds():
    """Verify forward pass on synthetic grayscale batch produces valid probabilities."""
    import tensorflow as tf

    model = build_transfer_model(input_shape=(90, 90, 1))

    # Synthetic batch of 4 grayscale images in [0.0, 1.0]
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, 90, 90, 1)).astype(np.float32)
    output = model(dummy_input, training=False)

    assert output.shape == (4, 1)
    assert output.dtype == tf.float32

    output_np = output.numpy()
    assert np.all(output_np >= 0.0)
    assert np.all(output_np <= 1.0)


def test_transfer_model_save_and_load(tmp_path: Path):
    """Verify transfer learning model can be saved to and loaded from disk."""
    import tensorflow as tf

    model = build_transfer_model(input_shape=(90, 90, 1))
    save_path = tmp_path / "transfer_test.keras"

    model.save(str(save_path))
    assert save_path.exists()

    loaded_model = tf.keras.models.load_model(str(save_path))
    assert loaded_model.input_shape == (None, 90, 90, 1)
    assert loaded_model.output_shape == (None, 1)
    assert loaded_model.count_params() == model.count_params()
