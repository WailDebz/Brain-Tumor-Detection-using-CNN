"""CLI utility to generate qualitative Grad-CAM saliency visualizations for demonstration."""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRANSFER_MODEL_PATH
from src.explainability.gradcam import generate_gradcam_explanation


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate qualitative Grad-CAM saliency heatmaps and overlays for brain MRI images."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input brain MRI image.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=TRANSFER_MODEL_PATH,
        help=f"Path to the trained model artifact. Default: {TRANSFER_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "gradcam",
        help="Directory to save generated Grad-CAM figures.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        choices=[0, 1],
        help="Target class to explain: 1 (Tumor) or 0 (No Tumor). Default: explains predicted class.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="Target convolutional layer name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay blending weight for the heatmap. Default: 0.45",
    )
    return parser.parse_args()


def save_gradcam_figure_panel(
    result,
    output_path: Path,
) -> Path:
    """Save a 3-panel figure: Original Scan, Normalized Grad-CAM Heatmap, Superimposed Overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Original Image
    axes[0].imshow(result.original_image)
    axes[0].set_title("Input MRI Scan", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im_hm = axes[1].imshow(result.heatmap_resized, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Grad-CAM Heatmap\n(Target: {result.target_class_name})", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im_hm, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Superimposed Overlay
    axes[2].imshow(result.overlay_image)
    axes[2].set_title(
        f"Saliency Overlay ($\\alpha={0.45}$)\nPred: {result.predicted_label} ($p={result.predicted_probability:.4f}$)",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM Explainability: {result.model_name} (Layer: {result.target_layer_name})\n"
        "[Qualitative Research Visualization - Non-Diagnostic]",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> int:
    """Run Grad-CAM generation CLI."""
    args = parse_args()

    print("\n" + "=" * 64)
    print("BRAIN TUMOR DETECTION: GRAD-CAM EXPLAINABILITY")
    print("=" * 64)
    print(f"Input Image Path        : {args.image}")
    print(f"Model Artifact Path     : {args.model_path}")
    print(f"Output Directory        : {args.output_dir.resolve()}")
    print(f"Target Class            : {args.class_index if args.class_index is not None else 'Auto (Predicted Class)'}")
    print("-" * 64 + "\n")

    try:
        import tensorflow as tf

        if not args.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {args.model_path.resolve()}")

        # Resolve image path
        img_path = args.image
        if not img_path.is_absolute() and not img_path.exists():
            candidate = PROJECT_ROOT / img_path
            if candidate.exists():
                img_path = candidate

        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found at {img_path.resolve()}")

        # Load model
        model = tf.keras.models.load_model(str(args.model_path))

        # Generate explanation
        result = generate_gradcam_explanation(
            model=model,
            image=img_path,
            target_layer_name=args.target_layer,
            class_index=args.class_index,
            alpha=args.alpha,
        )

        image_stem = img_path.stem.replace(" ", "_")
        target_slug = "tumor" if result.target_class == 1 else "no_tumor"
        output_filename = f"gradcam_{result.model_name}_{image_stem}_{target_slug}.png"
        output_filepath = args.output_dir / output_filename

        save_gradcam_figure_panel(result, output_filepath)

        print("Grad-CAM Explanation Summary:")
        print(f"  - Model Evaluated      : {result.model_name}")
        print(f"  - Target Feature Layer : {result.target_layer_name}")
        print(f"  - Predicted Probability: {result.predicted_probability:.4f} ({result.predicted_label})")
        print(f"  - Target Class Saliency: {result.target_class_name}")
        print(f"  - Saved Visualization  : {output_filepath.resolve()}")
        print("\n[SUCCESS] Grad-CAM figure generated successfully.\n")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] Grad-CAM generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
