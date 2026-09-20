"""Pretrained transfer learning model architecture (MobileNetV2 feature extractor)."""

from typing import Tuple
from src.config import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    NUM_CHANNELS,
)


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], NUM_CHANNELS),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_reg: float = DEFAULT_L2_REGULARIZATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    fine_tune: bool = False,
    name: str = "brain_tumor_transfer_mobilenetv2",
):
    """Construct and compile a transfer learning model using an ImageNet-pretrained MobileNetV2 backbone.

    ARCHITECTURE & METHODOLOGY:
    1. Backbone Selection (MobileNetV2):
       - MobileNetV2 (Sandler et al., 2018) utilizes inverted residual blocks with linear bottlenecks.
       - Parameter efficiency: ~2.26M backbone parameters, making it computationally practical for
         CPU inference and small-sample regime experiments without prohibitive compute costs.
    2. Input Channel & Normalization Adaptation:
       - Input MRI slice tensor: shape (90, 90, 1) with float32 pixel intensities in [0.0, 1.0].
       - In-Graph Channel Adaptation: The single grayscale channel is replicated across 3 channels
         using `Concatenate(axis=-1)([x, x, x])` to feed the 3-channel pretrained backbone.
       - In-Graph Normalization: MobileNetV2 expects input range [-1.0, +1.0]. A `Rescaling(2.0, offset=-1.0)`
         layer transforms [0.0, 1.0] inputs to [-1.0, +1.0] without modifying external datasets.
    3. Frozen Feature Extraction Mode:
       - By default (`fine_tune=False`), all backbone weights are frozen (`backbone.trainable = False`).
       - Only the top classification head is trained, preserving generic ImageNet feature extractors
         and preventing catastrophic forgetting on small medical datasets (159 training samples).
    4. Classification Head:
       - `GlobalAveragePooling2D()` spatial aggregation (1280 features).
       - `Dense(64, ReLU, L2)` feature compression.
       - `Dropout(dropout_rate)` regularization.
       - `Dense(1, Sigmoid)` binary probability prediction.

    Args:
        input_shape: (height, width, channels) of input image. Defaults to (90, 90, 1).
        learning_rate: Initial Adam learning rate. Defaults to 0.001.
        l2_reg: L2 regularization penalty on classification head. Defaults to 1e-4.
        dropout_rate: Dropout rate in classification head. Defaults to 0.3.
        fine_tune: If True, backbone weights are trainable; if False, backbone is frozen.
        name: Model identifier name.

    Returns:
        Compiled `tf.keras.Model` instance.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Step 1: Define input layer
    inputs = layers.Input(shape=input_shape, name="input_mri_slice")

    # Step 2: Adapt single-channel grayscale to 3-channel RGB in-graph
    if input_shape[-1] == 1:
        x_3ch = layers.Concatenate(axis=-1, name="grayscale_to_rgb")([inputs, inputs, inputs])
    else:
        x_3ch = inputs

    # Step 3: Rescale from [0, 1] to [-1, 1] (MobileNetV2 expected range)
    x_scaled = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale_to_mobilenet_range")(x_3ch)

    # Step 4: Instantiate pretrained MobileNetV2 backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze or unfreeze backbone
    backbone.trainable = fine_tune

    # Run forward pass through backbone
    # In feature extraction mode, ensure training=False so BatchNorm running stats remain frozen
    features = backbone(x_scaled, training=fine_tune)

    # Step 5: Feature aggregation & classification head
    pooled = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    dense = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_features",
    )(pooled)
    dropout = layers.Dropout(dropout_rate, name="dropout_head")(dense)
    outputs = layers.Dense(1, activation="sigmoid", name="output_probability")(dropout)

    # Construct model
    model = models.Model(inputs=inputs, outputs=outputs, name=name)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    return model
