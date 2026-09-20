from pathlib import Path
import sys
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DECISION_THRESHOLD,
    PRIMARY_APP_MODEL_PATH,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.explainability.gradcam import GradCAMResult, generate_gradcam_explanation
from src.inference.predictor import load_detector_model


@st.cache_resource
def load_cached_model(model_path: Path = PRIMARY_APP_MODEL_PATH) -> Optional[tf.keras.Model]:
    """Load and cache the primary Keras model artifact.

    Args:
        model_path: Path to the .keras model artifact.

    Returns:
        Loaded tf.keras.Model or None if loading fails.
    """
    try:
        return load_detector_model(model_path)
    except FileNotFoundError:
        st.error(
            f"Model artifact not found at: `{model_path.resolve()}`.\n\n"
            "Please ensure that the trained model exists in the `models/` directory."
        )
        return None
    except Exception as exc:
        st.error(f"Failed to load model artifact: {exc}")
        return None


def validate_uploaded_image(uploaded_file) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Validate that an uploaded file is a readable, non-empty image.

    Args:
        uploaded_file: Streamlit UploadedFile object or file path.

    Returns:
        Tuple of (PIL.Image.Image or None, error_message or None).
    """
    if uploaded_file is None:
        return None, "No file provided."

    try:
        img = Image.open(uploaded_file)
        # Verify basic image dimensions
        if img.width == 0 or img.height == 0:
            return None, "Uploaded image has invalid zero-dimensional bounds."

        # Convert palette/LA images to standardized RGB/RGBA for safe processing
        if img.mode not in {"RGB", "RGBA", "L"}:
            img = img.convert("RGB")

        return img, None
    except Exception as err:
        return None, f"Could not decode uploaded file as a valid image: {err}"


def format_prediction_output(
    probability: float,
    threshold: float = DECISION_THRESHOLD,
) -> Tuple[str, float]:
    """Format numerical probability into binary class label and percentage.

    Args:
        probability: Continuous prediction score in [0.0, 1.0].
        threshold: Decision threshold for positive tumor class.

    Returns:
        Tuple of (predicted_label, probability_percentage).
    """
    label = "Tumor" if probability >= threshold else "No Tumor"
    pct = float(probability * 100.0)
    return label, pct


def main() -> None:
    """Streamlit application UI entrypoint."""
    st.set_page_config(
        page_title="Brain MRI Tumor Classification",
        page_icon="🧠",
        layout="centered",
    )

    # 1. Header & Overview
    st.title("Brain MRI Tumor Classification")
    st.caption(
        "Demonstration of an experimental deep-learning classification pipeline and "
        "Grad-CAM visual explainability for axial brain MRI scans."
    )

    # 2. Model Status Banner
    st.info(
        "**Model:** MobileNetV2 (ImageNet-pretrained feature extractor) | "
        "**Architecture:** Frozen backbone with trained classification head (Experimental, Non-Diagnostic)"
    )

    # Load model (cached)
    model = load_cached_model(PRIMARY_APP_MODEL_PATH)

    # 3. Image Upload Section
    st.subheader("1. Upload Brain MRI Scan")
    uploaded_file = st.file_uploader(
        "Select an axial brain MRI image (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Upload a 2D axial slice in JPEG or PNG format.",
    )

    if uploaded_file is None:
        st.text("Please upload an image file to generate predictions and Grad-CAM saliency explanations.")
    else:
        # Validate uploaded image
        image, error_msg = validate_uploaded_image(uploaded_file)
        if error_msg is not None or image is None:
            st.error(error_msg or "Invalid image file.")
            return

        if model is None:
            st.warning("Model is not loaded. Cannot perform inference.")
            return

        # 4. Inference & Grad-CAM Explanation
        try:
            with st.spinner("Processing image and generating Grad-CAM explainability heatmap..."):
                result: GradCAMResult = generate_gradcam_explanation(
                    model=model,
                    image=image,
                    alpha=0.45,
                )
        except Exception as exc:
            st.error(
                f"Inference error: Could not process image ({exc}). "
                "Please ensure the input is a valid brain MRI scan."
            )
            return

        # 5. Prediction & Model Output Score
        st.subheader("2. Model Classification & Score")
        col1, col2 = st.columns(2)

        with col1:
            if result.predicted_label == "Tumor":
                st.error(f"**Predicted Class:** {result.predicted_label}")
            else:
                st.success(f"**Predicted Class:** {result.predicted_label}")

        with col2:
            prob_pct = result.predicted_probability * 100.0
            st.metric(
                label="Model Predicted Probability (Tumor)",
                value=f"{prob_pct:.1f}%",
            )

        st.caption(
            f"Decision threshold: {DECISION_THRESHOLD:.2f} (Scores $\\ge 0.50$ classified as Tumor, $< 0.50$ as No Tumor). "
            "This value represents the model's experimental output score for this image."
        )

        # 6. Grad-CAM Saliency Explanation
        st.subheader("3. Grad-CAM Saliency Explanation")
        st.write(
            f"Visual heatmap showing spatial regions contributing to the **{result.target_class_name}** score."
        )

        # Convert float heatmap to colored jet colormap for clean UI rendering
        heatmap_uint8 = np.uint8(255.0 * result.heatmap_resized)
        colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_hm, col_ov = st.columns(3)
        with col_img:
            st.image(result.original_image, caption="1. Input MRI Scan", use_container_width=True)
        with col_hm:
            st.image(colored_heatmap_rgb, caption="2. Grad-CAM Heatmap", use_container_width=True)
        with col_ov:
            st.image(result.overlay_image, caption="3. Saliency Overlay", use_container_width=True)

        st.caption(
            "**Note on Saliency Interpretation:** Highlighted regions indicate spatial features that contributed "
            "positively to the model's output score. Saliency heatmaps do NOT represent medical segmentation, "
            "tumor margin delineation, or confirmed pathology."
        )

    # 7. About the Model & Methodology
    with st.expander("About the Model & Methodology"):
        st.markdown(
            """
            - **Task:** Binary classification of 2D axial brain MRI slices (Tumor vs. No Tumor).
            - **Input Preprocessing:** Grayscale conversion, spatial resizing to $90 \\times 90$ pixels, pixel intensity normalization to $[0.0, 1.0]$.
            - **Architecture:** MobileNetV2 pretrained on ImageNet-1k, adapted with an in-graph 3-channel expander and range scaler.
            - **Training Mode:** Frozen feature extraction (2,257,984 frozen backbone parameters, 82,049 trainable classification head parameters).
            - **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) computed on the final feature layer (`out_relu`).
            - **Dataset:** Small research cohort (228 unique image files after exact SHA-256 duplicate removal).
            - **Metadata Limitation:** Patient IDs and DICOM headers are unavailable in the source dataset; patient-level independence is not established.
            """
        )

    # 8. Non-Diagnostic Disclaimer
    st.markdown("---")
    st.warning(
        "**Non-Diagnostic Research Disclaimer:**\n\n"
        "Educational / research prototype. This application is not a medical device and must not be used "
        "for diagnosis, patient screening, or clinical decision-making. Model outputs are experimental "
        "and may be incorrect."
    )


if __name__ == "__main__":
    main()

