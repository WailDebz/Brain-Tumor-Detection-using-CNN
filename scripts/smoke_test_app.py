"""Smoke test script for Streamlit application components and workflows."""

import io
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import tensorflow as tf

from app.streamlit_app import (
    format_prediction_output,
    load_cached_model,
    validate_uploaded_image,
)
from src.config import PRIMARY_APP_MODEL_PATH
from src.explainability.gradcam import generate_gradcam_explanation

print("=== Starting Streamlit Application Smoke Test ===")

# 1. Model Loading
print("\n[1] Testing Model Loading...")
model = load_cached_model(PRIMARY_APP_MODEL_PATH)
assert model is not None, "Failed to load primary application model."
print(f"  Model loaded successfully: {model.name} (Total params: {model.count_params()})")

# 2. RGB Image Upload & Inference
print("\n[2] Testing RGB Image Inference & Grad-CAM...")
rgb_path = Path("data/raw/yes/Y1.jpg")
assert rgb_path.exists(), f"Image not found: {rgb_path}"
with open(rgb_path, "rb") as f:
    buf_rgb = io.BytesIO(f.read())

img_rgb, err_rgb = validate_uploaded_image(buf_rgb)
assert err_rgb is None and img_rgb is not None, f"Validation failed: {err_rgb}"

result_rgb = generate_gradcam_explanation(model=model, image=img_rgb, alpha=0.45)
label_rgb, pct_rgb = format_prediction_output(result_rgb.predicted_probability)
print(f"  RGB Image Prediction: {label_rgb} (Model predicted probability: {pct_rgb:.1f}%)")
print(f"  Grad-CAM Saliency generated: Heatmap shape {result_rgb.heatmap_resized.shape}, Overlay shape {result_rgb.overlay_image.shape}")
assert result_rgb.predicted_label == "Tumor"
assert result_rgb.target_class == 1

# 3. Grayscale Image Upload & Inference
print("\n[3] Testing Grayscale Image Inference & Grad-CAM...")
gray_img_pil = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
buf_gray = io.BytesIO()
gray_img_pil.save(buf_gray, format="PNG")
buf_gray.seek(0)

img_gray, err_gray = validate_uploaded_image(buf_gray)
assert err_gray is None and img_gray is not None

result_gray = generate_gradcam_explanation(model=model, image=img_gray, alpha=0.45)
label_gray, pct_gray = format_prediction_output(result_gray.predicted_probability)
print(f"  Grayscale Image Prediction: {label_gray} (Model predicted probability: {pct_gray:.1f}%)")

# 4. RGBA Image Upload & Inference
print("\n[4] Testing RGBA Image Inference & Grad-CAM...")
rgba_img_pil = Image.fromarray(np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode="RGBA")
buf_rgba = io.BytesIO()
rgba_img_pil.save(buf_rgba, format="PNG")
buf_rgba.seek(0)

img_rgba, err_rgba = validate_uploaded_image(buf_rgba)
assert err_rgba is None and img_rgba is not None

result_rgba = generate_gradcam_explanation(model=model, image=img_rgba, alpha=0.45)
label_rgba, pct_rgba = format_prediction_output(result_rgba.predicted_probability)
print(f"  RGBA Image Prediction: {label_rgba} (Model predicted probability: {pct_rgba:.1f}%)")

# 5. Invalid File Error Handling
print("\n[5] Testing Invalid Image Handling...")
buf_invalid = io.BytesIO(b"corrupted binary data that is not an image")
img_inv, err_inv = validate_uploaded_image(buf_invalid)
assert img_inv is None and err_inv is not None
print(f"  Graceful validation error caught: '{err_inv}'")

# 6. Non-Tumor Demonstration Image
print("\n[6] Testing Non-Tumor Scan Inference & Negative-Class Grad-CAM...")
no_tumor_path = Path("data/raw/no/1 no.jpeg")
if no_tumor_path.exists():
    with open(no_tumor_path, "rb") as f:
        buf_no = io.BytesIO(f.read())
    img_no, err_no = validate_uploaded_image(buf_no)
    assert err_no is None
    result_no = generate_gradcam_explanation(model=model, image=img_no, alpha=0.45)
    label_no, pct_no = format_prediction_output(result_no.predicted_probability)
    print(f"  No-Tumor Scan Prediction: {label_no} (Model predicted probability: {pct_no:.1f}%)")
    print(f"  Target Class Explained: {result_no.target_class_name}")
    assert result_no.target_class == 0

print("\n=== ALL STREAMLIT COMPONENT SMOKE TESTS PASSED SUCCESSFULLY ===")
