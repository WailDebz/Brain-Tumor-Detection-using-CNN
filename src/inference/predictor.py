"""Model loader and inference utilities for brain tumor detection."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.config import DECISION_THRESHOLD, DEFAULT_MODEL_PATH
from src.inference.preprocessing import preprocess_image


def load_detector_model(model_path: Optional[Path] = None):
    """Load the trained Keras brain tumor detector model.

    Args:
        model_path: Path to the .keras model artifact. If None, uses DEFAULT_MODEL_PATH.

    Returns:
        Loaded tf.keras.Model instance.

    Raises:
        FileNotFoundError: If the specified model artifact does not exist on disk.
    """
    import tensorflow as tf

    resolved_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
    resolved_path = Path(resolved_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at expected location: {resolved_path.resolve()}.\n"
            "Please ensure the model file is placed in the 'models/' directory."
        )

    model = tf.keras.models.load_model(str(resolved_path))
    return model


def predict_tumor(
    model,
    image: Union[Image.Image, np.ndarray],
    threshold: float = DECISION_THRESHOLD,
) -> Tuple[float, str]:
    """Perform tumor classification inference on an input image.

    Args:
        model: Loaded Keras model.
        image: Raw input image (PIL Image or NumPy array).
        threshold: Decision boundary for positive class. Defaults to 0.5.

    Returns:
        Tuple of (probability_score, predicted_label), where probability_score is
        in [0.0, 1.0] and label is either 'Tumor' or 'No Tumor'.
    """
    preprocessed = preprocess_image(image)
    raw_prediction = model.predict(preprocessed, verbose=0)
    probability = float(raw_prediction[0][0]) if raw_prediction.ndim == 2 else float(raw_prediction[0])

    label = "Tumor" if probability > threshold else "No Tumor"
    return probability, label
