"""Environment and dependency verification script."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

output_lines = []

def log(msg: str):
    print(msg)
    output_lines.append(msg)

log(f"Python Executable: {sys.executable}")
log(f"Python Version: {sys.version}")

try:
    import numpy as np
    log(f"NumPy Version: {np.__version__}")
except ImportError as e:
    log(f"NumPy Import Failed: {e}")

try:
    import cv2
    log(f"OpenCV Version: {cv2.__version__}")
except ImportError as e:
    log(f"OpenCV Import Failed: {e}")

try:
    import PIL
    from PIL import Image
    log(f"Pillow Version: {PIL.__version__}")
except ImportError as e:
    log(f"Pillow Import Failed: {e}")

try:
    import sklearn
    log(f"Scikit-Learn Version: {sklearn.__version__}")
except ImportError as e:
    log(f"Scikit-Learn Import Failed: {e}")

try:
    import matplotlib
    log(f"Matplotlib Version: {matplotlib.__version__}")
except ImportError as e:
    log(f"Matplotlib Import Failed: {e}")

try:
    import streamlit
    log(f"Streamlit Version: {streamlit.__version__}")
except ImportError as e:
    log(f"Streamlit Import Failed: {e}")

try:
    import pytest
    log(f"Pytest Version: {pytest.__version__}")
except ImportError as e:
    log(f"Pytest Import Failed: {e}")

try:
    import tensorflow as tf
    import keras
    log(f"TensorFlow Version: {tf.__version__}")
    log(f"Keras Version: {keras.__version__}")
    log(f"Physical Devices: {tf.config.list_physical_devices()}")
except ImportError as e:
    log(f"TensorFlow Import Failed: {e}")

# Step 6 verification: Model instantiations
log("\n--- Model Verification ---")
try:
    from src.models.cnn import build_cnn_model
    model = build_cnn_model(input_shape=(90, 90, 1))
    log(f"Corrected CNN model instantiated successfully. Name: {model.name}, Params: {model.count_params()}")
except Exception as e:
    log(f"Failed to instantiate corrected CNN model: {e}")

try:
    model_path = Path("models/brain_tumor_detector.keras")
    if model_path.exists():
        import tensorflow as tf
        orig_model = tf.keras.models.load_model(str(model_path))
        log(f"Historical model loaded successfully from {model_path}. Name: {orig_model.name}, Params: {orig_model.count_params()}")
    else:
        log(f"Historical model not found at {model_path}")
except Exception as e:
    log(f"Historical model load result: {e}")



