"""Evaluation metrics and diagnostic visualization utilities."""

from src.evaluation.evaluator import (
    evaluate_model_on_manifest,
    evaluate_with_validation_threshold_tuning,
)
from src.evaluation.metrics import (
    EvaluationReport,
    apply_decision_threshold,
    calculate_accuracy,
    calculate_average_precision,
    calculate_f1_score,
    calculate_precision,
    calculate_roc_auc,
    calculate_sensitivity,
    calculate_specificity,
    compute_evaluation_report,
    extract_confusion_matrix_counts,
    find_optimal_threshold_on_validation,
)
from src.evaluation.visualizer import (
    generate_all_evaluation_figures,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)

__all__ = [
    "EvaluationReport",
    "apply_decision_threshold",
    "calculate_accuracy",
    "calculate_average_precision",
    "calculate_f1_score",
    "calculate_precision",
    "calculate_roc_auc",
    "calculate_sensitivity",
    "calculate_specificity",
    "compute_evaluation_report",
    "evaluate_model_on_manifest",
    "evaluate_with_validation_threshold_tuning",
    "extract_confusion_matrix_counts",
    "find_optimal_threshold_on_validation",
    "generate_all_evaluation_figures",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
]

