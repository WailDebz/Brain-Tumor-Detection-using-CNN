"""Gradient-weighted Class Activation Mapping (Grad-CAM) for brain MRI classification.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.

IMPORTANT NON-DIAGNOSTIC NOTICE:
    Grad-CAM heatmaps provide qualitative visualizations of spatial regions in the
    input image that contributed positively to the model's output score.
    Heatmaps do NOT constitute medical segmentation, diagnostic validation,
    or ground-truth tumor margin delineation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import DEFAULT_IMAGE_SIZE, PROJECT_ROOT
from src.data.preprocessing import preprocess_for_inference, preprocess_image_array


@dataclass
class GradCAMResult:
    """Container for Grad-CAM explainability outputs."""

    original_image: np.ndarray  # (H, W, 3) uint8 RGB image
    heatmap_raw: np.ndarray     # (H_feat, W_feat) float32 feature-level heatmap
    heatmap_resized: np.ndarray # (H, W) float32 normalized heatmap in [0.0, 1.0]
    overlay_image: np.ndarray   # (H, W, 3) uint8 RGB blended visualization
    predicted_probability: float # Model output score in [0.0, 1.0]
    predicted_label: str        # "Tumor" or "No Tumor"
    target_class: int           # 1 (Tumor) or 0 (No Tumor)
    target_class_name: str      # Name of class being explained
    target_layer_name: str      # Layer used for activation extraction
    model_name: str             # Identifier of the evaluated model


def find_target_conv_layer(model: Any, explicit_layer_name: Optional[str] = None) -> Tuple[Any, str]:
    """Identify the target convolutional feature layer for Grad-CAM.

    Hierarchy:
    1. If `explicit_layer_name` is provided, search model and nested submodels.
    2. For MobileNetV2 transfer models: looks for 'out_relu' or 'Conv_1' inside the nested backbone.
    3. For custom CNN baseline: looks for 'relu3', 'conv3_128', or the last Conv2D/Activation layer.

    Args:
        model: tf.keras.Model instance.
        explicit_layer_name: Optional explicit layer identifier.

    Returns:
        Tuple of (target_layer_object, target_layer_name).

    Raises:
        ValueError: If no suitable convolutional layer can be identified.
    """
    import tensorflow as tf

    # Check for nested MobileNetV2 backbone
    backbone = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    if explicit_layer_name is not None:
        # Check outer model
        try:
            return model.get_layer(explicit_layer_name), explicit_layer_name
        except (ValueError, KeyError):
            pass

        # Check backbone if nested
        if backbone is not None:
            try:
                return backbone.get_layer(explicit_layer_name), explicit_layer_name
            except (ValueError, KeyError):
                pass

        raise ValueError(
            f"Specified target layer '{explicit_layer_name}' not found in model '{model.name}' or its backbone."
        )

    # Auto-detection: MobileNetV2
    if backbone is not None:
        for candidate in ["out_relu", "Conv_1", "block_16_project"]:
            try:
                layer = backbone.get_layer(candidate)
                return layer, candidate
            except (ValueError, KeyError):
                continue

    # Auto-detection: Custom CNN or general models
    for candidate in ["relu3", "conv3_128", "conv3", "conv2_64"]:
        try:
            layer = model.get_layer(candidate)
            return layer, candidate
        except (ValueError, KeyError):
            continue

    # Fallback: search backwards for last Conv2D or Activation with 4D output
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.ReLU, tf.keras.layers.Activation)):
            return layer, layer.name

    raise ValueError(f"Could not automatically identify a convolutional layer for model '{model.name}'.")


def compute_gradcam_heatmap(
    model: Any,
    preprocessed_tensor: np.ndarray,
    target_layer_name: Optional[str] = None,
    class_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """Compute 2D Grad-CAM heatmap via TensorFlow GradientTape.

    MATHEMATICAL FORMULATION:
    1. Forward pass: compute activations A of target layer and prediction p = sigma(z).
    2. Target score definition:
       - For class 1 (Tumor): y = p (probability of tumor presence)
       - For class 0 (No Tumor): y = (1.0 - p) (probability of tumor absence)
    3. Gradients: compute d(y) / d(A^k) for each feature map k.
    4. Channel importance weights: alpha_k = (1 / Z) * sum_{i,j} (d(y) / d(A_{i,j}^k)) (GAP of gradients).
    5. Localization map: L = ReLU(sum_k alpha_k * A^k).
    6. Safe normalization: L / max(L) (returns zero matrix if max(L) == 0).

    Args:
        model: Loaded tf.keras.Model instance.
        preprocessed_tensor: (1, 90, 90, 1) float32 preprocessed input array.
        target_layer_name: Optional name of the convolutional layer.
        class_index: 1 for Tumor explanation, 0 for No Tumor explanation.

    Returns:
        Tuple of (feature_level_heatmap, normalized_2d_heatmap, probability_score, resolved_layer_name).
    """
    import tensorflow as tf

    target_layer_obj, resolved_name = find_target_conv_layer(model, target_layer_name)

    # Check whether model contains a nested backbone (e.g. MobileNetV2)
    has_nested_backbone = any("mobilenetv2" in l.name.lower() for l in model.layers)

    if has_nested_backbone:
        backbone = next(l for l in model.layers if "mobilenetv2" in l.name.lower())
        backbone_extractor = tf.keras.Model(
            inputs=backbone.input,
            outputs=[target_layer_obj.output, backbone.output],
            name="gradcam_backbone_extractor",
        )

        with tf.GradientTape() as tape:
            # 1. In-graph channel adaptation and scaling
            x_input = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)
            x_rgb = tf.concat([x_input, x_input, x_input], axis=-1)
            x_scaled = (x_rgb * 2.0) - 1.0

            # 2. Extract feature activations and backbone output
            target_activations, backbone_out = backbone_extractor(x_scaled, training=False)
            tape.watch(target_activations)

            # 3. Forward pass through classification head
            x = model.get_layer("global_avg_pool")(backbone_out)
            x = model.get_layer("dense_features")(x)
            x = model.get_layer("dropout_head")(x, training=False)
            predictions = model.get_layer("output_probability")(x)

            prob = predictions[:, 0]
            target_score = prob if class_index == 1 else (1.0 - prob)

        grads = tape.gradient(target_score, target_activations)

    else:
        # Sequential or standard feed-forward model (Custom CNN)
        with tf.GradientTape() as tape:
            x = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)
            target_activations = None

            for layer in model.layers:
                x = layer(x, training=False)
                if layer.name == target_layer_obj.name:
                    target_activations = x
                    tape.watch(target_activations)

            predictions = x
            prob = predictions[:, 0]
            target_score = prob if class_index == 1 else (1.0 - prob)

        if target_activations is None:
            raise ValueError(f"Target layer {target_layer_obj.name} was not encountered during forward pass.")

        grads = tape.gradient(target_score, target_activations)

    # Global Average Pooling of gradients over spatial dimensions
    # Shape: (1, 1, 1, channels)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)

    # Weighted linear combination of feature maps: sum_k (alpha_k * A^k)
    cam = tf.reduce_sum(weights * target_activations, axis=-1)

    # Apply ReLU to retain features that have a positive influence on target class
    cam = tf.maximum(cam, 0.0)

    # Extract single sample 2D array
    cam_raw = cam[0].numpy().astype(np.float32)

    # Safe normalization: scale to [0.0, 1.0] without NaN
    max_val = np.max(cam_raw)
    if max_val > 0.0 and np.isfinite(max_val):
        cam_norm = cam_raw / max_val
    else:
        cam_norm = np.zeros_like(cam_raw, dtype=np.float32)

    prob_value = float(prob[0].numpy()) if hasattr(prob[0], "numpy") else float(prob[0])
    return cam_raw, cam_norm, prob_value, resolved_name


def overlay_gradcam_heatmap(
    original_image: Union[Image.Image, np.ndarray, str, Path],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlay a 2D Grad-CAM heatmap onto the original RGB image.

    Args:
        original_image: Source image as Path, PIL Image, or NumPy uint8 array.
        heatmap: 2D float32 normalized heatmap array in [0.0, 1.0].
        alpha: Overlay blend weight for heatmap in [0.0, 1.0]. Defaults to 0.4.
        colormap: OpenCV colormap constant (defaults to cv2.COLORMAP_JET).

    Returns:
        Tuple of (original_rgb_uint8, resized_heatmap_float32, overlaid_rgb_uint8).
    """
    # 1. Load/format original image to RGB uint8
    if isinstance(original_image, (str, Path)):
        path = Path(original_image)
        if not path.is_absolute() and not path.exists():
            candidate = PROJECT_ROOT / path
            if candidate.exists():
                path = candidate
        with Image.open(path) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    elif isinstance(original_image, Image.Image):
        img_rgb = np.array(original_image.convert("RGB"), dtype=np.uint8)
    elif isinstance(original_image, np.ndarray):
        if original_image.ndim == 2:
            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.ndim == 3:
            if original_image.shape[2] == 4:
                img_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
            elif original_image.shape[2] == 1:
                img_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = original_image.copy()
        else:
            raise ValueError(f"Unsupported image array dimensions: {original_image.ndim}D")
        if img_rgb.dtype != np.uint8:
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255.0).astype(np.uint8)
            else:
                img_rgb = img_rgb.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image type: {type(original_image)}")

    h_target, w_target = img_rgb.shape[:2]

    # 2. Resize heatmap to match original image spatial dimensions
    heatmap_resized = cv2.resize(heatmap, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

    # 3. Convert normalized heatmap to RGB colormap
    heatmap_uint8 = np.uint8(255.0 * heatmap_resized)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 4. Superimpose heatmap onto original image
    clamped_alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = cv2.addWeighted(
        colored_heatmap_rgb,
        clamped_alpha,
        img_rgb,
        1.0 - clamped_alpha,
        0.0,
    )

    return img_rgb, heatmap_resized, overlay


def generate_gradcam_explanation(
    model: Any,
    image: Union[Image.Image, np.ndarray, str, Path],
    target_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
    alpha: float = 0.4,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> GradCAMResult:
    """Generate complete end-to-end Grad-CAM explanation for an image.

    Args:
        model: Loaded tf.keras.Model instance.
        image: Input image (Path, PIL Image, or NumPy array).
        target_layer_name: Optional target convolutional layer name.
        class_index: Class to explain: 1 (Tumor), 0 (No Tumor), or None (explains predicted class).
        alpha: Overlay blending factor in [0.0, 1.0]. Defaults to 0.4.
        target_size: Model input spatial resolution. Defaults to (90, 90).

    Returns:
        GradCAMResult populated with original, heatmap, overlay, scores, and metadata.
    """
    # 1. Preprocess input for model forward pass
    preprocessed_tensor = preprocess_for_inference(image, target_size=target_size)

    # 2. If class_index is not specified, run inference first to explain the model's predicted class
    if class_index is None:
        raw_pred = model(preprocessed_tensor, training=False).numpy().ravel()[0]
        chosen_class = 1 if raw_pred >= 0.5 else 0
    else:
        chosen_class = int(class_index)

    # 3. Compute Grad-CAM feature map and normalized heatmap
    cam_raw, cam_norm, probability, resolved_layer = compute_gradcam_heatmap(
        model=model,
        preprocessed_tensor=preprocessed_tensor,
        target_layer_name=target_layer_name,
        class_index=chosen_class,
    )

    # 4. Generate visual overlay
    orig_rgb, heatmap_resized, overlay_rgb = overlay_gradcam_heatmap(
        original_image=image,
        heatmap=cam_norm,
        alpha=alpha,
    )

    predicted_label = "Tumor" if probability >= 0.5 else "No Tumor"
    target_class_name = "Tumor (Positive Class)" if chosen_class == 1 else "No Tumor (Negative Class)"
    model_name = getattr(model, "name", "model")

    return GradCAMResult(
        original_image=orig_rgb,
        heatmap_raw=cam_raw,
        heatmap_resized=heatmap_resized,
        overlay_image=overlay_rgb,
        predicted_probability=probability,
        predicted_label=predicted_label,
        target_class=chosen_class,
        target_class_name=target_class_name,
        target_layer_name=resolved_layer,
        model_name=model_name,
    )
