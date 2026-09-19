"""Reproducibility utilities for deterministic random state initialization."""

import os
import random
from typing import Optional
import numpy as np

from src.config import RANDOM_SEED


def set_random_seeds(seed: Optional[int] = RANDOM_SEED) -> int:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility.

    Applies deterministic random state to:
    1. Python `random` module.
    2. NumPy `np.random` generator.
    3. Python hash seed (`PYTHONHASHSEED` environment variable).
    4. TensorFlow `tf.random` and Keras backend via `tf.keras.utils.set_random_seed`.

    IMPORTANT REPRODUCIBILITY LIMITATIONS:
    - While random number generators (weight initialization, batch shuffling, data splits)
      are made deterministic, bit-level identical floating-point results across different
      hardware architectures (e.g. differing GPU CUDA cores, cuDNN versions, CPU vector
      extensions, or parallel reduction trees) cannot be guaranteed.
    - Deterministic operation flags in deep learning frameworks can introduce runtime
      performance trade-offs on specific GPU operators.

    Args:
        seed: Integer random seed. Defaults to `src.config.RANDOM_SEED` (42).

    Returns:
        The integer seed applied.
    """
    effective_seed = seed if seed is not None else 42

    # 1. Python built-in random
    random.seed(effective_seed)

    # 2. Python hash seed
    os.environ["PYTHONHASHSEED"] = str(effective_seed)

    # 3. NumPy RNG
    np.random.seed(effective_seed)

    # 4. TensorFlow & Keras
    try:
        import tensorflow as tf

        tf.random.set_seed(effective_seed)
        tf.keras.utils.set_random_seed(effective_seed)
    except ImportError:
        # Graceful fallback if TensorFlow is not yet loaded in non-DL context
        pass

    return effective_seed
