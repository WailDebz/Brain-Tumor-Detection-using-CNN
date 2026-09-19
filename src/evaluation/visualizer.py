"""Diagnostic visualization utilities for medical classification models."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn import metrics as sk_metrics

from src.config import (
    CLASS_LABELS,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_PR_CURVE_PATH,
    DEFAULT_ROC_CURVE_PATH,
    REPORTS_FIGURES_DIR,
)
from src.evaluation.metrics import (
    calculate_average_precision,
    calculate_roc_auc,
    calculate_sensitivity,
    calculate_specificity,
    extract_confusion_matrix_counts,
)


def _setup_matplotlib():
    """Ensure non-interactive backend is configured for headless environments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path] = DEFAULT_CONFUSION_MATRIX_PATH,
    class_names: Tuple[str, str] = ("No Tumor", "Tumor"),
    model_name: str = "Baseline CNN",
    split_name: str = "Test",
) -> Path:
    """Generate and save a clear, annotated medical confusion matrix heatmap.

    Layout:
      - Rows: Ground Truth / Actual Condition
      - Columns: Model Prediction

    Annotations show raw sample counts and within-class percentages.

    Args:
        y_true: 1D array of ground truth labels (0 or 1).
        y_pred: 1D array of binary predictions (0 or 1).
        output_path: Target PNG file path.
        class_names: Display labels for (class 0, class 1). Defaults to ("No Tumor", "Tumor").
        model_name: Name of evaluated model.
        split_name: Partition name ('Test', 'Validation', etc.).

    Returns:
        Path of the saved PNG figure.
    """
    plt = _setup_matplotlib()
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tp, tn, fp, fn = extract_confusion_matrix_counts(y_true, y_pred)
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot matrix using clean colormap
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Class totals for percentages
    row_totals = cm.sum(axis=1)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            row_total = row_totals[i] if row_totals[i] > 0 else 1
            pct = (count / row_total) * 100
            label_text = f"{count}\n({pct:.1f}%)"
            text_color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, label_text, ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    # Configure axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    ax.set_xlabel("Predicted Classification", fontsize=12, labelpad=10, fontweight="semibold")
    ax.set_ylabel("Actual Ground Truth", fontsize=12, labelpad=10, fontweight="semibold")
    ax.set_title(
        f"Confusion Matrix: {model_name}\n({split_name} Set Partition)",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(target_path, dpi=180)
    plt.close(fig)

    return target_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path] = DEFAULT_ROC_CURVE_PATH,
    model_name: str = "Baseline CNN",
    split_name: str = "Test",
) -> Path:
    """Generate and save the Receiver Operating Characteristic (ROC) curve.

    Plots:
      - Empirical ROC Curve (True Positive Rate vs. False Positive Rate)
      - Chance Diagonal (FPR = TPR, AUC = 0.50)
      - Annotated ROC-AUC value

    Args:
        y_true: 1D array of ground truth labels (0 or 1).
        y_prob: 1D array of predicted continuous probabilities in [0.0, 1.0].
        output_path: Target PNG file path.
        model_name: Model name identifier.
        split_name: Split partition name ('Test', etc.).

    Returns:
        Path of the saved PNG figure.
    """
    plt = _setup_matplotlib()
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_prob, dtype=np.float32).ravel()

    auc_score = calculate_roc_auc(yt, yp)
    auc_label = f"ROC Curve (AUC = {auc_score:.4f})" if auc_score is not None else "ROC Curve (AUC = N/A)"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if len(np.unique(yt)) >= 2:
        fpr, tpr, _ = sk_metrics.roc_curve(yt, yp)
        ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=auc_label)
    else:
        ax.text(
            0.5,
            0.5,
            "ROC Curve unavailable:\nOnly one class present in ground truth.",
            ha="center",
            va="center",
            fontsize=11,
            color="red",
        )

    # Random chance baseline
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    ax.set_title(
        f"Receiver Operating Characteristic (ROC)\n{model_name} - {split_name} Set",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(target_path, dpi=180)
    plt.close(fig)

    return target_path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path] = DEFAULT_PR_CURVE_PATH,
    model_name: str = "Baseline CNN",
    split_name: str = "Test",
) -> Path:
    """Generate and save the Precision-Recall (PR) curve.

    Particularly informative for imbalanced clinical imaging distributions.
    Plots:
      - Precision vs. Recall curve across all possible decision thresholds
      - Baseline prevalence line (Positives / Total)
      - Annotated Average Precision (PR-AUC) score

    Args:
        y_true: 1D array of ground truth labels (0 or 1).
        y_prob: 1D array of predicted continuous probabilities in [0.0, 1.0].
        output_path: Target PNG file path.
        model_name: Model name identifier.
        split_name: Split partition name.

    Returns:
        Path of the saved PNG figure.
    """
    plt = _setup_matplotlib()
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_prob, dtype=np.float32).ravel()

    ap_score = calculate_average_precision(yt, yp)
    ap_label = f"PR Curve (AP = {ap_score:.4f})" if ap_score is not None else "PR Curve (AP = N/A)"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if len(np.unique(yt)) >= 2:
        precision, recall, _ = sk_metrics.precision_recall_curve(yt, yp)
        ax.plot(recall, precision, color="#2ca02c", lw=2.5, label=ap_label)

        # Baseline horizontal line representing prevalence of tumor class
        prevalence = float(np.sum(yt == 1) / len(yt))
        ax.axhline(
            y=prevalence,
            color="gray",
            linestyle="--",
            lw=1.5,
            label=f"Prevalence Baseline ({prevalence * 100:.1f}%)",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "PR Curve unavailable:\nOnly one class present in ground truth.",
            ha="center",
            va="center",
            fontsize=11,
            color="red",
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="semibold")
    ax.set_title(
        f"Precision-Recall (PR) Curve\n{model_name} - {split_name} Set",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(target_path, dpi=180)
    plt.close(fig)

    return target_path


def generate_all_evaluation_figures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Union[str, Path] = REPORTS_FIGURES_DIR,
    model_name: str = "Baseline CNN",
    split_name: str = "Test",
) -> Dict[str, Path]:
    """Generate and save all diagnostic evaluation plots.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Continuous predicted probabilities in [0.0, 1.0].
        y_pred: Binary predictions (0 or 1).
        figures_dir: Destination directory for figures.
        model_name: Model identifier string.
        split_name: Split partition name.

    Returns:
        Dictionary mapping figure names ('confusion_matrix', 'roc_curve', 'pr_curve') to saved Paths.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm_path = fig_dir / f"confusion_matrix_{split_name.lower()}.png"
    roc_path = fig_dir / f"roc_curve_{split_name.lower()}.png"
    pr_path = fig_dir / f"pr_curve_{split_name.lower()}.png"

    paths = {
        "confusion_matrix": plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            output_path=cm_path,
            model_name=model_name,
            split_name=split_name,
        ),
        "roc_curve": plot_roc_curve(
            y_true=y_true,
            y_prob=y_prob,
            output_path=roc_path,
            model_name=model_name,
            split_name=split_name,
        ),
        "pr_curve": plot_precision_recall_curve(
            y_true=y_true,
            y_prob=y_prob,
            output_path=pr_path,
            model_name=model_name,
            split_name=split_name,
        ),
    }

    return paths
