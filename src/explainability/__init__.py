"""Visual explainability and saliency mapping module (Grad-CAM)."""

from src.explainability.gradcam import (
    GradCAMResult,
    compute_gradcam_heatmap,
    generate_gradcam_explanation,
    overlay_gradcam_heatmap,
)

__all__ = [
    "GradCAMResult",
    "compute_gradcam_heatmap",
    "generate_gradcam_explanation",
    "overlay_gradcam_heatmap",
]
