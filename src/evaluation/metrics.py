"""Comprehensive medical image classification metrics and evaluation reporting."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from src.config import DECISION_THRESHOLD


@dataclass
class EvaluationReport:
    """Structured container for model evaluation metrics and metadata."""

    model_name: str
    dataset_split: str
    threshold: float
    threshold_strategy: str
    total_samples: int
    positive_samples: int
    negative_samples: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    sensitivity: float  # Recall / True Positive Rate
    specificity: float  # True Negative Rate
    f1_score: float
    roc_auc: Optional[float]
    average_precision: Optional[float]  # PR-AUC

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation report to dictionary."""
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert evaluation report to a single-row pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save report to JSON format."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    def save_csv(self, output_path: Union[str, Path]) -> Path:
        """Save report to CSV format."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(target, index=False)
        return target

    def summary(self) -> str:
        """Format a human-readable clinical evaluation summary."""
        auc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A (single class)"
        pr_auc_str = (
            f"{self.average_precision:.4f}"
            if self.average_precision is not None
            else "N/A (single class)"
        )
        lines = [
            "=" * 64,
            f"HELD-OUT EVALUATION REPORT: {self.model_name.upper()}",
            "=" * 64,
            f"Dataset Split            : {self.dataset_split.upper()}",
            f"Decision Threshold       : {self.threshold:.4f} ({self.threshold_strategy})",
            f"Total Evaluated Samples  : {self.total_samples}",
            f"Ground Truth Prevalence  : {self.positive_samples} Tumor (1) | {self.negative_samples} No Tumor (0)",
            "-" * 64,
            "Confusion Matrix:",
            f"  True Positives  (TP)   : {self.tp:4d}  (Correctly identified Tumor)",
            f"  True Negatives  (TN)   : {self.tn:4d}  (Correctly identified No Tumor)",
            f"  False Positives (FP)   : {self.fp:4d}  (Type I Error - False Alarm)",
            f"  False Negatives (FN)   : {self.fn:4d}  (Type II Error - Missed Tumor)",
            "-" * 64,
            "Performance Metrics:",
            f"  Accuracy               : {self.accuracy * 100:6.2f}%",
            f"  Sensitivity (Recall)   : {self.sensitivity * 100:6.2f}%  [TP / (TP + FN)]",
            f"  Specificity (TNR)      : {self.specificity * 100:6.2f}%  [TN / (TN + FP)]",
            f"  Precision (PPV)        : {self.precision * 100:6.2f}%  [TP / (TP + FP)]",
            f"  F1-Score (Harmonic)    : {self.f1_score * 100:6.2f}%",
            f"  ROC-AUC                : {auc_str}",
            f"  PR-AUC (Avg Precision) : {pr_auc_str}",
            "=" * 64,
        ]
        return "\n".join(lines)


def apply_decision_threshold(
    y_prob: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> np.ndarray:
    """Convert continuous probability predictions in [0.0, 1.0] to binary classes {0, 1}.

    Args:
        y_prob: 1D array of predicted probabilities.
        threshold: Decision boundary for positive class. Defaults to 0.5.

    Returns:
        1D integer NumPy array of shape (N,) containing binary predictions (0 or 1).
    """
    probs = np.asarray(y_prob, dtype=np.float32).ravel()
    return (probs >= threshold).astype(int)


def extract_confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Calculate counts of True Positives, True Negatives, False Positives, and False Negatives.

    Class definitions:
        Positive (1) = Tumor
        Negative (0) = No Tumor

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).

    Returns:
        Tuple of (tp, tn, fp, fn) as integers.
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    return tp, tn, fp, fn


def calculate_sensitivity(tp: int, fn: int) -> float:
    """Compute Sensitivity (also known as Recall or True Positive Rate).

    Formula: Sensitivity = TP / (TP + FN)
    Measures the proportion of actual tumors correctly identified.

    Args:
        tp: True positive count.
        fn: False negative count.

    Returns:
        Sensitivity in [0.0, 1.0]. Returns 0.0 if no actual positives exist (TP + FN == 0).
    """
    total_positives = tp + fn
    return float(tp / total_positives) if total_positives > 0 else 0.0


def calculate_specificity(tn: int, fp: int) -> float:
    """Compute Specificity (also known as True Negative Rate).

    Formula: Specificity = TN / (TN + FP)
    Measures the proportion of healthy/non-tumor scans correctly identified.

    Args:
        tn: True negative count.
        fp: False positive count.

    Returns:
        Specificity in [0.0, 1.0]. Returns 0.0 if no actual negatives exist (TN + FP == 0).
    """
    total_negatives = tn + fp
    return float(tn / total_negatives) if total_negatives > 0 else 0.0


def calculate_precision(tp: int, fp: int) -> float:
    """Compute Precision (also known as Positive Predictive Value).

    Formula: Precision = TP / (TP + FP)
    Measures the proportion of positive tumor predictions that are genuinely positive.

    Args:
        tp: True positive count.
        fp: False positive count.

    Returns:
        Precision in [0.0, 1.0]. Returns 0.0 if no positive predictions exist (TP + FP == 0).
    """
    total_predicted_positives = tp + fp
    return float(tp / total_predicted_positives) if total_predicted_positives > 0 else 0.0


def calculate_f1_score(precision: float, sensitivity: float) -> float:
    """Compute F1-Score (harmonic mean of Precision and Sensitivity).

    Formula: F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

    Args:
        precision: Precision value in [0.0, 1.0].
        sensitivity: Sensitivity value in [0.0, 1.0].

    Returns:
        F1-Score in [0.0, 1.0]. Returns 0.0 if sum is 0.0.
    """
    denom = precision + sensitivity
    return float(2.0 * (precision * sensitivity) / denom) if denom > 0.0 else 0.0


def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute overall Classification Accuracy.

    Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        tp: True positive count.
        tn: True negative count.
        fp: False positive count.
        fn: False negative count.

    Returns:
        Accuracy in [0.0, 1.0]. Returns 0.0 if total count is 0.
    """
    total = tp + tn + fp + fn
    return float((tp + tn) / total) if total > 0 else 0.0


def calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability scores in [0.0, 1.0].

    Returns:
        ROC-AUC score in [0.0, 1.0], or None if only one unique class is present in y_true.
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_prob, dtype=np.float32).ravel()

    if len(np.unique(yt)) < 2:
        return None

    try:
        score = sk_metrics.roc_auc_score(yt, yp)
        return float(score)
    except Exception:
        return None


def calculate_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute Area Under the Precision-Recall Curve (PR-AUC / Average Precision).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability scores in [0.0, 1.0].

    Returns:
        Average Precision score in [0.0, 1.0], or None if only one class is present in y_true.
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_prob, dtype=np.float32).ravel()

    if len(np.unique(yt)) < 2:
        return None

    try:
        score = sk_metrics.average_precision_score(yt, yp)
        return float(score)
    except Exception:
        return None


def compute_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
    threshold_strategy: str = "fixed_default_0.5",
    model_name: str = "brain_tumor_baseline_cnn",
    dataset_split: str = "test",
) -> EvaluationReport:
    """Compute full suite of binary classification metrics and package into EvaluationReport.

    Args:
        y_true: 1D array of ground truth labels (0 = No Tumor, 1 = Tumor).
        y_prob: 1D array of predicted probabilities in [0.0, 1.0].
        threshold: Decision threshold for positive classification.
        threshold_strategy: Descriptor for how the threshold was determined.
        model_name: Name/identifier of the model being evaluated.
        dataset_split: Split partition name ('test', 'val', 'train').

    Returns:
        EvaluationReport populated with all clinical and statistical metrics.
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_prob, dtype=np.float32).ravel()

    # Binarize predictions
    y_pred = apply_decision_threshold(yp, threshold=threshold)

    # Confusion matrix elements
    tp, tn, fp, fn = extract_confusion_matrix_counts(yt, y_pred)

    # Derived metrics
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    precision = calculate_precision(tp, fp)
    sensitivity = calculate_sensitivity(tp, fn)
    specificity = calculate_specificity(tn, fp)
    f1 = calculate_f1_score(precision, sensitivity)
    roc_auc = calculate_roc_auc(yt, yp)
    avg_prec = calculate_average_precision(yt, yp)

    pos_count = int(np.sum(yt == 1))
    neg_count = int(np.sum(yt == 0))

    return EvaluationReport(
        model_name=model_name,
        dataset_split=dataset_split,
        threshold=float(threshold),
        threshold_strategy=threshold_strategy,
        total_samples=len(yt),
        positive_samples=pos_count,
        negative_samples=neg_count,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        sensitivity=sensitivity,
        specificity=specificity,
        f1_score=f1,
        roc_auc=roc_auc,
        average_precision=avg_prec,
    )


def find_optimal_threshold_on_validation(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "youden_j",
    num_thresholds: int = 101,
) -> Tuple[float, float]:
    """Find the optimal decision threshold using ONLY the validation dataset partition.

    CRITICAL METHODOLOGICAL SAFETY NOTICE:
    - This function MUST be executed exclusively on the VALIDATION split.
    - NEVER call this function using test set labels or predictions.
    - Optimizing a decision threshold on the test set is a form of data snooping / post-hoc
      leakage that invalidates the held-out evaluation benchmark.
    - The optimal threshold chosen here must be frozen and then applied once to the test set.

    Supported optimization metrics:
    - "youden_j": Youden's J statistic (Sensitivity + Specificity - 1). Maximizes diagnostic power.
    - "f1": F1-Score (Harmonic mean of Precision and Sensitivity).
    - "sensitivity_at_high_specificity": Maximizes Sensitivity subject to Specificity >= 0.85.

    Args:
        y_true_val: Ground truth labels for the validation split.
        y_prob_val: Predicted probabilities for the validation split.
        metric: Criterion to optimize ('youden_j', 'f1', 'sensitivity_at_high_specificity').
        num_thresholds: Number of discrete threshold steps to search in [0.01, 0.99].

    Returns:
        Tuple of (optimal_threshold, best_metric_score).
    """
    yt = np.asarray(y_true_val, dtype=int).ravel()
    yp = np.asarray(y_prob_val, dtype=np.float32).ravel()

    candidate_thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_score = -1.0

    for thresh in candidate_thresholds:
        y_pred = apply_decision_threshold(yp, threshold=thresh)
        tp, tn, fp, fn = extract_confusion_matrix_counts(yt, y_pred)

        sens = calculate_sensitivity(tp, fn)
        spec = calculate_specificity(tn, fp)
        prec = calculate_precision(tp, fp)
        f1 = calculate_f1_score(prec, sens)

        if metric == "youden_j":
            # Youden's J index = Sensitivity + Specificity - 1 (range: [-1.0, 1.0])
            score = sens + spec - 1.0
        elif metric == "f1":
            score = f1
        elif metric == "sensitivity_at_high_specificity":
            # Clinical priority: high sensitivity while maintaining acceptable specificity
            score = sens if spec >= 0.85 else -1.0
        else:
            raise ValueError(f"Unsupported threshold optimization metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    return best_threshold, best_score
