"""Unit tests for online data augmentation layers."""

import numpy as np
import pytest

from src.data.augment import build_augmentation_layer


def test_build_augmentation_layer_structure():
    """Verify augmentation layer creates expected Keras Sequential pipeline."""
    import tensorflow as tf

    aug_layer = build_augmentation_layer(image_size=(90, 90), seed=42)
    assert isinstance(aug_layer, tf.keras.Sequential)
    assert len(aug_layer.layers) >= 3

    layer_names = [layer.name for layer in aug_layer.layers]
    assert any("flip" in name for name in layer_names)
    assert any("rotation" in name for name in layer_names)
    assert any("translation" in name for name in layer_names)


def test_augmentation_forward_pass_shape_and_dtype():
    """Verify augmentation forward pass preserves tensor shape and data bounds."""
    import tensorflow as tf

    aug_layer = build_augmentation_layer(image_size=(90, 90), seed=42)

    # Create synthetic batch: 4 images of shape (90, 90, 1) with values in [0.0, 1.0]
    dummy_batch = np.random.uniform(0.0, 1.0, size=(4, 90, 90, 1)).astype(np.float32)
    dummy_tensor = tf.convert_to_tensor(dummy_batch)

    # In training mode, augmentation is applied
    augmented = aug_layer(dummy_tensor, training=True)

    assert augmented.shape == (4, 90, 90, 1)
    assert augmented.dtype == tf.float32
    # Check that tensor remains within valid intensity bounds [0.0, 1.0]
    assert tf.reduce_min(augmented).numpy() >= 0.0
    assert tf.reduce_max(augmented).numpy() <= 1.0


def test_augmentation_identity_in_inference_mode():
    """Verify that in non-training mode, the layer passes images through unchanged."""
    import tensorflow as tf

    aug_layer = build_augmentation_layer(image_size=(90, 90), seed=42)
    dummy_batch = np.random.uniform(0.0, 1.0, size=(2, 90, 90, 1)).astype(np.float32)
    dummy_tensor = tf.convert_to_tensor(dummy_batch)

    # In inference mode (training=False), output must match input
    output = aug_layer(dummy_tensor, training=False)
    np.testing.assert_allclose(dummy_tensor.numpy(), output.numpy(), atol=1e-5)
