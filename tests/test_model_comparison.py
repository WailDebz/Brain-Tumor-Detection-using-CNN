"""Unit tests for multi-model benchmark comparison reporting."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.evaluation.comparison import ModelComparisonEntry, generate_comparison_report


def test_model_comparison_entry_to_dict():
    """Verify ModelComparisonEntry converts to dictionary correctly."""
    entry = ModelComparisonEntry(
        model_name="test_model",
        architecture_type="CNN",
        split="Test (35 images)",
        test_set_usage="Exploratory Benchmark",
        total_parameters=1000,
        trainable_parameters=900,
        non_trainable_parameters=100,
        best_validation_loss=0.45,
        best_validation_epoch=5,
        test_threshold=0.5,
        test_threshold_strategy="fixed",
        accuracy=0.85,
        precision=0.80,
        sensitivity=0.90,
        specificity=0.80,
        f1_score=0.85,
        roc_auc=0.92,
        average_precision=0.91,
        tp=18,
        tn=16,
        fp=4,
        fn=2,
        notes="Test note",
    )

    d = entry.to_dict()
    assert isinstance(d, dict)
    assert d["model_name"] == "test_model"
    assert d["accuracy"] == 0.85
    assert d["tp"] == 18
    assert d["test_set_usage"] == "Exploratory Benchmark"


def test_generate_comparison_report(tmp_path: Path):
    """Verify generate_comparison_report writes valid JSON and CSV artifacts."""
    entry1 = ModelComparisonEntry(
        model_name="custom_cnn",
        architecture_type="Custom CNN",
        split="Test (35 images)",
        test_set_usage="Exploratory Benchmark",
        total_parameters=101889,
        trainable_parameters=101441,
        non_trainable_parameters=448,
        best_validation_loss=0.6780,
        best_validation_epoch=5,
        test_threshold=0.5,
        test_threshold_strategy="fixed_default_0.5",
        accuracy=0.6286,
        precision=0.6286,
        sensitivity=1.0,
        specificity=0.0,
        f1_score=0.7719,
        roc_auc=0.7902,
        average_precision=0.8089,
        tp=22,
        tn=0,
        fp=13,
        fn=0,
    )
    entry2 = ModelComparisonEntry(
        model_name="transfer_mobilenetv2_frozen",
        architecture_type="MobileNetV2 (Frozen)",
        split="Test (35 images)",
        test_set_usage="Exploratory Benchmark",
        total_parameters=2340033,
        trainable_parameters=82049,
        non_trainable_parameters=2257984,
        best_validation_loss=0.5500,
        best_validation_epoch=8,
        test_threshold=0.5,
        test_threshold_strategy="fixed_default_0.5",
        accuracy=0.8000,
        precision=0.8200,
        sensitivity=0.8500,
        specificity=0.7500,
        f1_score=0.8350,
        roc_auc=0.8800,
        average_precision=0.8900,
        tp=19,
        tn=9,
        fp=4,
        fn=3,
    )

    json_path = tmp_path / "comparison.json"
    csv_path = tmp_path / "comparison.csv"

    saved_json, saved_csv, df = generate_comparison_report(
        entries=[entry1, entry2],
        output_json_path=json_path,
        output_csv_path=csv_path,
    )

    assert saved_json.exists()
    assert saved_csv.exists()
    assert len(df) == 2

    # Verify JSON content
    with open(saved_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["model_name"] == "custom_cnn"
        assert data[1]["model_name"] == "transfer_mobilenetv2_frozen"

    # Verify CSV content
    loaded_df = pd.read_csv(saved_csv)
    assert len(loaded_df) == 2
    assert "model_name" in loaded_df.columns
    assert "accuracy" in loaded_df.columns
