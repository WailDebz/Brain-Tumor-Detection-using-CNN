"""Pretrained transfer learning model architecture (MobileNetV2 feature extractor & fine-tuning)."""

from pathlib import Path
from typing import Optional, Tuple, Union
from src.config import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_FINETUNING_LEARNING_RATE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_UNFREEZE_FROM_LAYER,
    NUM_CHANNELS,
)


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], NUM_CHANNELS),
    learning_rate: Optional[float] = None,
    l2_reg: float = DEFAULT_L2_REGULARIZATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    fine_tune: bool = False,
    unfreeze_from_layer: int = DEFAULT_UNFREEZE_FROM_LAYER,
    weights_path: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
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
    3. Transfer Learning Modes:
       - Feature Extraction (`fine_tune=False`):
         All backbone weights are frozen (`backbone.trainable = False`). Only the top classification head
         is trained with initial learning rate 1e-3.
       - Controlled Fine-Tuning (`fine_tune=True`):
         The lower backbone layers (0 to unfreeze_from_layer-1) remain frozen. Only the top inverted residual
         blocks (from unfreeze_from_layer onwards) and classification head are trainable.
         CRITICAL BATCHNORMALIZATION RULE: All BatchNormalization layers across the entire backbone
         remain strictly non-trainable (`trainable = False`) to prevent destructive updates to running
         mean and variance on small medical batches.
         Compiled with a smaller learning rate (default: 1e-5).
    4. Classification Head:
       - `GlobalAveragePooling2D()` spatial aggregation (1280 features).
       - `Dense(64, ReLU, L2)` feature compression.
       - `Dropout(dropout_rate)` regularization.
       - `Dense(1, Sigmoid)` binary probability prediction.

    Args:
        input_shape: (height, width, channels) of input image. Defaults to (90, 90, 1).
        learning_rate: Initial Adam learning rate. Defaults to 1e-3 (frozen) or 1e-5 (fine_tune).
        l2_reg: L2 regularization penalty on classification head. Defaults to 1e-4.
        dropout_rate: Dropout rate in classification head. Defaults to 0.3.
        fine_tune: If True, top backbone layers are trainable; if False, entire backbone is frozen.
        unfreeze_from_layer: Layer index from which backbone layers become trainable when fine_tune=True.
            Default: 143 (block_16_expand, unfreezing Block 16 + final Conv_1).
        weights_path: Optional path to pre-trained weights (e.g. from frozen feature extractor) to load.
        name: Model identifier name.

    Returns:
        Compiled `tf.keras.Model` instance.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Determine default learning rate based on mode
    if learning_rate is None:
        effective_lr = DEFAULT_FINETUNING_LEARNING_RATE if fine_tune else DEFAULT_LEARNING_RATE
    else:
        effective_lr = learning_rate

    # Determine default model name
    default_name = "brain_tumor_transfer_mobilenetv2_finetuned" if fine_tune else "brain_tumor_transfer_mobilenetv2_frozen"
    model_name = name or default_name

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

    # Configure layer trainability
    if not fine_tune:
        backbone.trainable = False
    else:
        backbone.trainable = True
        for i, layer in enumerate(backbone.layers):
            if i < unfreeze_from_layer:
                layer.trainable = False
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # Freeze BatchNormalization to protect running stats on small batches
                layer.trainable = False
            else:
                layer.trainable = True

    # Run forward pass through backbone
    features = backbone(x_scaled)

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
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # If weights path is provided, load initial weights
    if weights_path is not None:
        p = Path(weights_path)
        if p.exists():
            model.load_weights(str(p))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=effective_lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    return model
