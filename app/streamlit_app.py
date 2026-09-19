from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Resolve paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "brain_tumor_detector.keras"


@st.cache_resource
def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Optional[tf.keras.Model]:
    """Load the trained Keras model with caching.

    Args:
        model_path: Path to the .keras model artifact.

    Returns:
        Loaded tf.keras.Model or None if file does not exist.
    """
    if not model_path.exists():
        st.error(
            f"Model artifact not found at: `{model_path}`.\n\n"
            "Please ensure that `brain_tumor_detector.keras` is placed inside the `models/` directory."
        )
        return None
    try:
        model = tf.keras.models.load_model(str(model_path))
        return model
    except Exception as exc:
        st.error(f"Failed to load model from `{model_path}`: {exc}")
        return None


def preprocess_and_predict(image: Image.Image, model: tf.keras.Model) -> float:
    """Preprocess the input PIL image and run inference.

    Args:
        image: Uploaded PIL Image.
        model: Loaded Keras model.

    Returns:
        Predicted probability score as a float.
    """
    # Resize to 90x90
    resized = image.resize((90, 90))
    img_array = np.array(resized)

    # Convert to grayscale
    if img_array.ndim == 3:
        if img_array.shape[2] == 4:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Expand channel and batch dimensions
    img_input = np.expand_dims(gray, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    # Normalize pixel values
    img_input = img_input.astype(np.float32) / 255.0

    # Predict probability
    raw_prediction = model.predict(img_input, verbose=0)
    probability = float(raw_prediction[0][0]) if raw_prediction.ndim == 2 else float(raw_prediction[0])
    return probability


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.write("# Brain Tumor Detection")

    model = load_model()

    file = st.file_uploader(
        "Upload your brain axis scan image here",
        type=["jpg", "png", "jpeg"],
    )

    if file is None:
        st.text("Please upload an image file")
        return

    # Open the uploaded image
    try:
        image = Image.open(file)
    except Exception as err:
        st.error(f"Error opening image file: {err}")
        return

    st.image(image, use_container_width=True)

    if model is None:
        st.warning("Model is not loaded. Cannot perform inference.")
        return

    # Make prediction
    probability = preprocess_and_predict(image, model)

    # Binary decision
    label = "Tumor" if probability > 0.5 else "No Tumor"

    # Display the result
    st.success(f"This image is indicating: {label}")


if __name__ == "__main__":
    main()

