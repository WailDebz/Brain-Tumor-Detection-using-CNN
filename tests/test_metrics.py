"""Unit tests for medical evaluation metrics, confusion matrix extraction, and edge cases."""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn import metrics as sk_metrics

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


def test_hand_crafted_perfect_classification():
    """Verify metrics for perfect synthetic predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.7, 0.9])
    y_pred = apply_decision_threshold(y_prob, threshold=0.5)

    tp, tn, fp, fn = extract_confusion_matrix_counts(y_true, y_pred)
    assert (tp, tn, fp, fn) == (2, 2, 0, 0)

    acc = calculate_accuracy(tp, tn, fp, fn)
    sens = calculate_sensitivity(tp, fn)
    spec = calculate_specificity(tn, fp)
    prec = calculate_precision(tp, fp)
    f1 = calculate_f1_score(prec, sens)
    roc_auc = calculate_roc_auc(y_true, y_prob)
    pr_auc = calculate_average_precision(y_true, y_prob)

    assert acc == pytest.approx(1.0)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert roc_auc == pytest.approx(1.0)
    assert pr_auc == pytest.approx(1.0)


def test_hand_crafted_mixed_classification():
    """Verify metrics on a known 8-sample mixture with 2 FP and 1 FN."""
    # 4 healthy (0), 4 tumors (1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.3, 0.7, 0.85, 0.9])
    # At threshold 0.5:
    # y_pred = [0, 0, 1, 1, 0, 1, 1, 1]
    # TN=2 (0,1), FP=2 (2,3), FN=1 (4), TP=3 (5,6,7)
    y_pred = apply_decision_threshold(y_prob, threshold=0.5)

    tp, tn, fp, fn = extract_confusion_matrix_counts(y_true, y_pred)
    assert (tp, tn, fp, fn) == (3, 2, 2, 1)

    acc = calculate_accuracy(tp, tn, fp, fn)
    sens = calculate_sensitivity(tp, fn)
    spec = calculate_specificity(tn, fp)
    prec = calculate_precision(tp, fp)
    f1 = calculate_f1_score(prec, sens)

    assert acc == pytest.approx(5 / 8)  # 0.625
    assert sens == pytest.approx(3 / 4)  # 0.75
    assert spec == pytest.approx(2 / 4)  # 0.50
    assert prec == pytest.approx(3 / 5)  # 0.60
    assert f1 == pytest.approx((2 * 0.60 * 0.75) / (0.60 + 0.75))  # 2/3 ≈ 0.6667

    # Compare with scikit-learn references
    sk_acc = sk_metrics.accuracy_score(y_true, y_pred)
    sk_prec = sk_metrics.precision_score(y_true, y_pred)
    sk_rec = sk_metrics.recall_score(y_true, y_pred)
    sk_f1 = sk_metrics.f1_score(y_true, y_pred)

    assert acc == pytest.approx(sk_acc)
    assert prec == pytest.approx(sk_prec)
    assert sens == pytest.approx(sk_rec)
    assert f1 == pytest.approx(sk_f1)


def test_zero_division_safety_all_negative_predictions():
    """Verify behavior when model predicts negative (0) for all instances."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])  # All below 0.5
    y_pred = apply_decision_threshold(y_prob, threshold=0.5)

    tp, tn, fp, fn = extract_confusion_matrix_counts(y_true, y_pred)
    assert (tp, tn, fp, fn) == (0, 2, 0, 2)

    # Precision should return 0.0 safely instead of ZeroDivisionError
    prec = calculate_precision(tp, fp)
    assert prec == 0.0

    sens = calculate_sensitivity(tp, fn)
    assert sens == 0.0

    f1 = calculate_f1_score(prec, sens)
    assert f1 == 0.0

    spec = calculate_specificity(tn, fp)
    assert spec == pytest.approx(1.0)


def test_zero_division_safety_all_positive_predictions():
    """Verify behavior when model predicts positive (1) for all instances."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.6, 0.7, 0.8, 0.9])  # All above 0.5
    y_pred = apply_decision_threshold(y_prob, threshold=0.5)

    tp, tn, fp, fn = extract_confusion_matrix_counts(y_true, y_pred)
    assert (tp, tn, fp, fn) == (2, 0, 2, 0)

    spec = calculate_specificity(tn, fp)
    assert spec == 0.0

    sens = calculate_sensitivity(tp, fn)
    assert sens == pytest.approx(1.0)

    prec = calculate_precision(tp, fp)
    assert prec == pytest.approx(0.5)


def test_single_class_ground_truth_returns_none_for_auc():
    """Verify ROC-AUC and Average Precision return None when only 1 class is present."""
    y_true_single = np.array([1, 1, 1, 1])
    y_prob = np.array([0.8, 0.9, 0.7, 0.85])

    roc_auc = calculate_roc_auc(y_true_single, y_prob)
    pr_auc = calculate_average_precision(y_true_single, y_prob)

    assert roc_auc is None
    assert pr_auc is None


def test_find_optimal_threshold_on_validation():
    """Verify validation threshold selection chooses optimal boundary."""
    y_true_val = np.array([0, 0, 0, 1, 1, 1])
    # Predictions where negative class is ~0.2-0.4 and positive class is ~0.6-0.9
    y_prob_val = np.array([0.2, 0.3, 0.4, 0.6, 0.8, 0.9])

    best_thresh, best_score = find_optimal_threshold_on_validation(
        y_true_val,
        y_prob_val,
        metric="youden_j",
    )

    assert 0.4 < best_thresh <= 0.6
    assert best_score == pytest.approx(1.0)  # Perfect separation possible at 0.5


def test_evaluation_report_serialization(tmp_path: Path):
    """Verify EvaluationReport dataclass methods (dict, dataframe, JSON, CSV)."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9])

    report = compute_evaluation_report(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        threshold_strategy="fixed_test",
        model_name="test_cnn",
        dataset_split="test",
    )

    assert isinstance(report, EvaluationReport)
    assert report.total_samples == 4
    assert report.tp == 2
    assert report.tn == 2
    assert report.accuracy == 1.0

    # JSON save/load
    json_path = tmp_path / "test_report.json"
    report.save_json(json_path)
    assert json_path.exists()
    with open(json_path, "r") as f:
        loaded = json.load(f)
        assert loaded["accuracy"] == 1.0
        assert loaded["model_name"] == "test_cnn"

    # CSV save/load
    csv_path = tmp_path / "test_report.csv"
    report.save_csv(csv_path)
    assert csv_path.exists()
    df = report.to_dataframe()
    assert len(df) == 1
    assert df["accuracy"].iloc[0] == 1.0

    # Summary text
    summary_text = report.summary()
    assert "HELD-OUT EVALUATION REPORT: TEST_CNN" in summary_text
    assert "100.00%" in summary_text
