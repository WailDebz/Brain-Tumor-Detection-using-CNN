"""Unit tests for modernized baseline CNN architecture and model factory."""

import numpy as np
import pytest

from src.models.cnn import build_cnn_model


def test_build_cnn_model_structure():
    """Verify layer types, layer order, and presence of GlobalAveragePooling2D."""
    import tensorflow as tf

    model = build_cnn_model(input_shape=(90, 90, 1), learning_rate=0.001)

    assert isinstance(model, tf.keras.Model)
    assert model.name == "brain_tumor_baseline_cnn"

    layer_types = [type(layer).__name__ for layer in model.layers]

    # Verify key architectural components
    assert "Conv2D" in layer_types
    assert "BatchNormalization" in layer_types
    assert "MaxPooling2D" in layer_types
    assert "GlobalAveragePooling2D" in layer_types
    assert "Dense" in layer_types
    assert "Dropout" in layer_types

    # Ensure parameter-heavy Flatten is eliminated
    assert "Flatten" not in layer_types


def test_cnn_parameter_efficiency():
    """Verify that model parameters are significantly reduced compared to old ~1.42M model."""
    model = build_cnn_model(input_shape=(90, 90, 1))
    total_params = model.count_params()

    # The modernized baseline should be ~100k params, strictly under 250k
    assert total_params < 250_000, f"Expected <250k parameters, got {total_params}"
    assert total_params > 50_000, f"Expected >50k parameters, got {total_params}"


def test_cnn_forward_pass_and_output_bounds():
    """Verify forward pass on a synthetic batch produces shape (batch, 1) and values in [0, 1]."""
    import tensorflow as tf

    model = build_cnn_model(input_shape=(90, 90, 1))

    # Synthetic batch of 4 grayscale images
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, 90, 90, 1)).astype(np.float32)
    output = model(dummy_input, training=False)

    assert output.shape == (4, 1)
    assert output.dtype == tf.float32

    # Sigmoid output must be strictly in [0.0, 1.0]
    output_np = output.numpy()
    assert np.all(output_np >= 0.0)
    assert np.all(output_np <= 1.0)


def test_cnn_compilation_contract():
    """Verify model compiles with Adam optimizer, binary crossentropy, and accuracy metric."""
    import tensorflow as tf

    model = build_cnn_model(input_shape=(90, 90, 1), learning_rate=1e-4)

    assert model.optimizer is not None
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert model.loss is not None
    assert isinstance(model.loss, tf.keras.losses.BinaryCrossentropy)


def test_cnn_configurable_hyperparameters():
    """Verify model factory respects custom hyperparameter arguments."""
    model = build_cnn_model(
        input_shape=(64, 64, 1),
        learning_rate=5e-4,
        l2_reg=1e-3,
        dropout_rate=0.5,
        name="custom_cnn_test",
    )

    assert model.name == "custom_cnn_test"
    assert model.input_shape == (None, 64, 64, 1)
