"""Online training-only data augmentation pipeline using Keras layers."""

from typing import Optional, Tuple
from src.config import DEFAULT_IMAGE_SIZE, RANDOM_SEED


def build_augmentation_layer(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    seed: Optional[int] = RANDOM_SEED,
):
    """Build an online data augmentation Sequential layer for training.

    IMPORTANT METHODOLOGICAL NOTES:
    1. Online-Only: Transformations are computed on-the-fly in GPU/CPU memory per batch.
       No synthetic or augmented images are written to disk.
    2. Training-Only Constraint: This augmentation layer must ONLY be attached to the
       training pipeline. Validation and Test pipelines MUST evaluate exclusively on
       unaltered, deterministic original images.
    3. Experimental Notice: These affine and spatial transformations are standard computer
       vision regularization techniques and are NOT certified or clinically validated
       anatomical deformations.

    Transformations included:
    - Random horizontal flip (axial brain MRI symmetry)
    - Small random rotation (+/- 8% or ~28 degrees)
    - Minor random translation (+/- 5% height/width)
    - Minor random zoom (+/- 5%)

    Args:
        image_size: (width, height) target image dimensions.
        seed: Random seed for deterministic reproducibility.

    Returns:
        tf.keras.Sequential containing Keras preprocessing/augmentation layers.
    """
    import tensorflow as tf

    target_w, target_h = image_size

    augmentation_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed, name="aug_random_flip"),
            tf.keras.layers.RandomRotation(
                factor=0.08, fill_mode="constant", fill_value=0.0, seed=seed, name="aug_random_rotation"
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="constant",
                fill_value=0.0,
                seed=seed,
                name="aug_random_translation",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="constant",
                fill_value=0.0,
                seed=seed,
                name="aug_random_zoom",
            ),
        ],
        name="online_data_augmentation",
    )

    return augmentation_layer
