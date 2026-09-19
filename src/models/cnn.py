"""Modernized baseline Convolutional Neural Network for brain tumor classification."""

from typing import Optional, Tuple
from src.config import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    NUM_CHANNELS,
)


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], NUM_CHANNELS),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_reg: float = DEFAULT_L2_REGULARIZATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    name: str = "brain_tumor_baseline_cnn",
):
    """Construct and compile a modernized baseline 2D CNN for binary tumor classification.

    ARCHITECTURE DESIGN & MOTIVATION:
    The original prototype suffered from severe parameter bottlenecking:
    - Original: Flatten(10,368) -> Dense(128) concentrated 1,327,232 parameters (93.5% of model)
      into a single dense layer, encouraging training sample memorization on small cohorts.
    - Modernized Baseline: Replaces Flatten with `GlobalAveragePooling2D()`, reducing parameter
      count by ~93% (~102k params vs ~1.42M params) while enhancing spatial translation invariance.
    - Incorporates `BatchNormalization()` after convolutional stages for gradient stabilization.
    - Employs modest L2 kernel regularization and Dropout (0.3) in the dense classification head.

    Layer Hierarchy:
    1. Input: (height, width, channels) = (90, 90, 1)
    2. Conv Block 1: Conv2D(32, 3x3, same) -> BatchNorm -> ReLU -> MaxPool(2x2) -> (45, 45, 32)
    3. Conv Block 2: Conv2D(64, 3x3, same) -> BatchNorm -> ReLU -> MaxPool(2x2) -> (22, 22, 64)
    4. Conv Block 3: Conv2D(128, 3x3, same) -> BatchNorm -> ReLU -> MaxPool(2x2) -> (11, 11, 128)
    5. Feature Aggregation: GlobalAveragePooling2D -> (128,)
    6. Classification Head: Dense(64, ReLU, L2) -> Dropout(dropout_rate) -> Dense(1, Sigmoid)

    Args:
        input_shape: (height, width, channels) input image dimensions. Defaults to (90, 90, 1).
        learning_rate: Initial Adam optimizer learning rate. Defaults to 0.001.
        l2_reg: L2 kernel regularization penalty. Defaults to 1e-4.
        dropout_rate: Dropout probability before the output unit. Defaults to 0.3.
        name: Model identifier name.

    Returns:
        Compiled `tf.keras.Model` instance ready for training.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Sequential model definition
    model = models.Sequential(
        [
            # Input specification
            layers.Input(shape=input_shape, name="input_mri_slice"),
            # Convolutional Block 1
            layers.Conv2D(
                32,
                kernel_size=(3, 3),
                padding="same",
                kernel_regularizer=regularizer,
                name="conv1_32",
            ),
            layers.BatchNormalization(name="bn1"),
            layers.Activation("relu", name="relu1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
            # Convolutional Block 2
            layers.Conv2D(
                64,
                kernel_size=(3, 3),
                padding="same",
                kernel_regularizer=regularizer,
                name="conv2_64",
            ),
            layers.BatchNormalization(name="bn2"),
            layers.Activation("relu", name="relu2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
            # Convolutional Block 3
            layers.Conv2D(
                128,
                kernel_size=(3, 3),
                padding="same",
                kernel_regularizer=regularizer,
                name="conv3_128",
            ),
            layers.BatchNormalization(name="bn3"),
            layers.Activation("relu", name="relu3"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
            # Spatial Feature Aggregation (Replaces parameter-heavy Flatten)
            layers.GlobalAveragePooling2D(name="global_avg_pool"),
            # Classification Head
            layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizer,
                name="dense_features",
            ),
            layers.Dropout(dropout_rate, name="dropout_head"),
            layers.Dense(1, activation="sigmoid", name="output_probability"),
        ],
        name=name,
    )

    # Compile with Adam and standard Binary Crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    return model
