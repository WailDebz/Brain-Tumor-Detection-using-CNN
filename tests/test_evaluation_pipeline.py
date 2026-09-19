"""Unit tests for evaluation pipeline, diagnostic plotting, and threshold tuning."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.evaluation.evaluator import (
    evaluate_model_on_manifest,
    evaluate_with_validation_threshold_tuning,
)
from src.evaluation.visualizer import (
    generate_all_evaluation_figures,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


@pytest.fixture
def synthetic_test_manifest_and_images(tmp_path: Path) -> Tuple[Path, pd.DataFrame]:
    """Create a temporary test directory with 6 synthetic images and a manifest CSV."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()

    records = []
    # 3 negative (0) and 3 positive (1) samples
    for i in range(6):
        label = 0 if i < 3 else 1
        class_name = "no" if label == 0 else "yes"
        file_path = img_dir / f"test_{class_name}_{i}.png"
        img = Image.new("RGB", (90, 90), color=(i * 35, 100, 150))
        img.save(file_path)

        records.append(
            {
                "filepath": str(file_path.resolve()),
                "filename": file_path.name,
                "class_name": class_name,
                "label": label,
                "file_hash": f"hash_{class_name}_{i}",
            }
        )

    df = pd.DataFrame(records)
    manifest_path = tmp_path / "test_manifest.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path, df


def test_visualizer_figure_generation(tmp_path: Path):
    """Verify visualizer generates confusion matrix, ROC curve, and PR curve figures."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.7, 0.9])
    y_pred = np.array([0, 0, 1, 1])

    figures_dir = tmp_path / "figures"
    fig_paths = generate_all_evaluation_figures(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        figures_dir=figures_dir,
        model_name="Test Baseline",
        split_name="Test",
    )

    assert "confusion_matrix" in fig_paths
    assert "roc_curve" in fig_paths
    assert "pr_curve" in fig_paths

    for name, p in fig_paths.items():
        assert p.exists()
        assert p.stat().st_size > 0


def test_evaluate_model_on_manifest_with_mock_model(
    synthetic_test_manifest_and_images: Tuple[Path, pd.DataFrame],
    tmp_path: Path,
):
    """Verify evaluate_model_on_manifest generates valid report, figures, and predictions."""
    manifest_path, df = synthetic_test_manifest_and_images

    # Lightweight mock Keras model returning predictable probabilities
    class MockKerasModel:
        def predict(self, dataset, verbose=0):
            # 6 predictions matching the 6 manifest samples
            # First 3 (label 0): 0.1, 0.2, 0.3 -> All predicted 0
            # Last 3 (label 1): 0.8, 0.9, 0.7 -> All predicted 1
            return np.array([[0.1], [0.2], [0.3], [0.8], [0.9], [0.7]], dtype=np.float32)

    mock_model = MockKerasModel()
    json_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "metrics.csv"
    fig_dir = tmp_path / "figs"

    report, fig_paths, pred_df = evaluate_model_on_manifest(
        model=mock_model,
        manifest=manifest_path,
        threshold=0.5,
        threshold_strategy="fixed_test",
        model_name="mock_model",
        split_name="test",
        metrics_json_path=json_path,
        metrics_csv_path=csv_path,
        figures_dir=fig_dir,
        verbose=0,
    )

    # 1. Verify Report
    assert report.total_samples == 6
    assert report.tp == 3
    assert report.tn == 3
    assert report.fp == 0
    assert report.fn == 0
    assert report.accuracy == 1.0
    assert report.sensitivity == 1.0
    assert report.specificity == 1.0

    # 2. Verify Output Files
    assert json_path.exists()
    assert csv_path.exists()
    assert len(fig_paths) == 3

    # 3. Verify Predictions DataFrame
    assert len(pred_df) == 6
    assert "predicted_probability" in pred_df.columns
    assert "predicted_binary" in pred_df.columns
    assert "is_correct" in pred_df.columns
    assert pred_df["is_correct"].all()


def test_evaluate_with_validation_threshold_tuning(
    synthetic_test_manifest_and_images: Tuple[Path, pd.DataFrame],
    tmp_path: Path,
):
    """Verify validation-based threshold tuning workflow."""
    manifest_path, df = synthetic_test_manifest_and_images

    class MockKerasModel:
        def predict(self, dataset, verbose=0):
            # Returns probabilities: [0.15, 0.25, 0.35, 0.65, 0.75, 0.85]
            return np.array([[0.15], [0.25], [0.35], [0.65], [0.75], [0.85]], dtype=np.float32)

    mock_model = MockKerasModel()

    report, fig_paths, pred_df, chosen_thresh = evaluate_with_validation_threshold_tuning(
        model=mock_model,
        val_manifest=manifest_path,
        test_manifest=manifest_path,
        metric_for_threshold="youden_j",
        model_name="mock_model",
        metrics_json_path=tmp_path / "tuned_metrics.json",
        metrics_csv_path=tmp_path / "tuned_metrics.csv",
        figures_dir=tmp_path / "tuned_figs",
        verbose=0,
    )

    assert 0.35 < chosen_thresh <= 0.65
    assert report.accuracy == 1.0
    assert report.threshold_strategy == "validation_tuned_youden_j"


def test_evaluate_missing_model_raises_error(tmp_path: Path):
    """Verify evaluate_model_on_manifest raises FileNotFoundError when model path is missing."""
    with pytest.raises(FileNotFoundError):
        evaluate_model_on_manifest(
            model=tmp_path / "non_existent_model.keras",
            manifest=tmp_path / "manifest.csv",
            verbose=0,
        )


def test_evaluate_empty_manifest_raises_error():
    """Verify evaluate_model_on_manifest raises ValueError when manifest is empty."""
    empty_df = pd.DataFrame(columns=["filepath", "label"])
    with pytest.raises(ValueError, match="contains 0 samples"):
        evaluate_model_on_manifest(
            model="dummy_model",
            manifest=empty_df,
            verbose=0,
        )
