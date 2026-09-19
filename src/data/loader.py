"""tf.data dataset builder for training, validation, and testing pipelines."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, PROJECT_ROOT, RANDOM_SEED
from src.data.augment import build_augmentation_layer
from src.data.ingestion import load_manifest


def _parse_and_preprocess_image(
    file_path: str,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
):
    """TensorFlow-native image decoding and normalization.

    Args:
        file_path: Tensor string containing image file path.
        target_size: (width, height) target image dimensions.

    Returns:
        Tensor of shape (height, width, 1) with dtype tf.float32 normalized to [0, 1].
    """
    import tensorflow as tf

    target_w, target_h = target_size

    # Read binary file
    raw_bytes = tf.io.read_file(file_path)

    # Decode image as 1-channel grayscale
    image = tf.io.decode_image(raw_bytes, channels=1, expand_animations=False)

    # Resize with bilinear interpolation
    image = tf.image.resize(image, [target_h, target_w], method="area")

    # Normalize to [0.0, 1.0]
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [target_h, target_w, 1])
    return image


def build_tf_dataset(
    manifest: Union[pd.DataFrame, str, Path],
    is_training: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    enable_augmentation: Optional[bool] = None,
    shuffle_buffer: int = 500,
    seed: int = RANDOM_SEED,
):
    """Construct an optimized tf.data.Dataset from a manifest.

    Pipeline specification:
    - Training pipeline: File paths -> Decode/Normalize -> Shuffle -> Batch -> Online Augmentation -> Prefetch.
    - Validation/Test pipeline: File paths -> Decode/Normalize -> Batch -> Prefetch (Deterministic, zero augmentation).

    Args:
        manifest: DataFrame or path to manifest CSV containing 'filepath' and 'label'.
        is_training: True for training set; False for validation and testing.
        batch_size: Number of samples per batch. Defaults to 32.
        target_size: Target (width, height). Defaults to (90, 90).
        enable_augmentation: If True, applies augmentation. If None, defaults to value of is_training.
        shuffle_buffer: Size of the shuffle buffer for training.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Configured tf.data.Dataset yielding (images, labels) batches.
    """
    import tensorflow as tf

    if isinstance(manifest, (str, Path)):
        df = load_manifest(manifest)
    else:
        df = manifest.copy()

    def _resolve_to_filesystem_path(p: str) -> str:
        path_obj = Path(p)
        if not path_obj.is_absolute() and not path_obj.exists():
            candidate = PROJECT_ROOT / path_obj
            if candidate.exists():
                return str(candidate.resolve())
        return str(path_obj.resolve())

    filepaths = [_resolve_to_filesystem_path(p) for p in df["filepath"].astype(str).values]
    labels = df["label"].astype(np.float32).values

    # Determine whether augmentation is applied
    should_augment = is_training if enable_augmentation is None else enable_augmentation

    # Create dataset from slices
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Map deterministic image loading and preprocessing
    def _map_fn(path, label):
        img = _parse_and_preprocess_image(path, target_size=target_size)
        return img, label

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), shuffle_buffer), seed=seed)

    # Batch dataset
    dataset = dataset.batch(batch_size)

    # Apply online augmentation layer to training batches only
    if should_augment:
        aug_layer = build_augmentation_layer(image_size=target_size, seed=seed)

        def _augment_batch(images, batch_labels):
            return aug_layer(images, training=True), batch_labels

        dataset = dataset.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch batches for optimal hardware pipeline utilization
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_dataset_splits_tf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    seed: int = RANDOM_SEED,
):
    """Construct train, validation, and test tf.data.Datasets in a single call.

    Args:
        train_df: Training manifest DataFrame.
        val_df: Validation manifest DataFrame.
        test_df: Test manifest DataFrame.
        batch_size: Batch size. Defaults to 32.
        target_size: Target (width, height). Defaults to (90, 90).
        seed: Random seed.

    Returns:
        Tuple of (train_ds, val_ds, test_ds).
    """
    train_ds = build_tf_dataset(
        train_df,
        is_training=True,
        batch_size=batch_size,
        target_size=target_size,
        seed=seed,
    )
    val_ds = build_tf_dataset(
        val_df,
        is_training=False,
        batch_size=batch_size,
        target_size=target_size,
        seed=seed,
    )
    test_ds = build_tf_dataset(
        test_df,
        is_training=False,
        batch_size=batch_size,
        target_size=target_size,
        seed=seed,
    )
    return train_ds, val_ds, test_ds
