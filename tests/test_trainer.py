"""Unit tests for the training pipeline and artifact generation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.training.trainer import (
    TrainingResult,
    plot_and_save_training_curves,
    train_baseline_model,
)
from src.utils.reproducibility import set_random_seeds


@pytest.fixture
def synthetic_training_dataset(tmp_path: Path) -> Path:
    """Create a minimal synthetic dataset directory with 12 images (6 'no', 6 'yes')."""
    data_dir = tmp_path / "raw_synthetic"
    no_dir = data_dir / "no"
    yes_dir = data_dir / "yes"

    no_dir.mkdir(parents=True)
    yes_dir.mkdir(parents=True)

    for i in range(6):
        arr_no = np.full((90, 90, 3), fill_value=i * 30, dtype=np.uint8)
        img_no = Image.fromarray(arr_no)
        img_no.save(no_dir / f"no_{i}.jpg")

        arr_yes = np.full((90, 90, 3), fill_value=(i + 1) * 35, dtype=np.uint8)
        img_yes = Image.fromarray(arr_yes)
        img_yes.save(yes_dir / f"yes_{i}.png")

    return data_dir


def test_set_random_seeds():
    """Verify seeding utility executes and returns applied seed."""
    applied_seed = set_random_seeds(123)
    assert applied_seed == 123


def test_plot_and_save_training_curves(tmp_path: Path):
    """Verify plot generation produces a valid PNG file."""
    mock_history = {
        "loss": [0.65, 0.45, 0.30],
        "val_loss": [0.68, 0.48, 0.35],
        "accuracy": [0.60, 0.80, 0.90],
        "val_accuracy": [0.55, 0.75, 0.85],
    }
    plot_path = tmp_path / "test_curves.png"
    saved_path = plot_and_save_training_curves(mock_history, plot_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_train_baseline_model_end_to_end(synthetic_training_dataset: Path, tmp_path: Path):
    """Verify full end-to-end training pipeline executes and saves artifacts."""
    model_path = tmp_path / "models" / "test_baseline.keras"
    history_json = tmp_path / "reports" / "test_history.json"
    history_csv = tmp_path / "reports" / "test_history.csv"
    plot_path = tmp_path / "reports" / "test_plot.png"

    # Train for 2 epochs on synthetic dataset
    model, result = train_baseline_model(
        data_source=synthetic_training_dataset,
        model_output_path=model_path,
        history_json_path=history_json,
        history_csv_path=history_csv,
        plot_output_path=plot_path,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        split_ratios=(0.60, 0.20, 0.20),
        early_stopping_patience=2,
        seed=42,
        verbose=0,
    )

    # 1. Check TrainingResult properties
    assert isinstance(result, TrainingResult)
    assert result.epochs_requested == 2
    assert result.epochs_completed == 2
    assert 0.0 <= result.best_val_accuracy <= 1.0
    assert result.best_val_loss > 0.0

    # 2. Check artifact files on disk
    assert model_path.exists()
    assert history_json.exists()
    assert history_csv.exists()
    assert plot_path.exists()

    # 3. Check JSON schema
    with open(history_json, "r") as f:
        data = json.load(f)
        assert "epochs_requested" in data
        assert "epochs_completed" in data
        assert "best_epoch" in data
        assert "best_val_loss" in data
        assert "history" in data
        assert "loss" in data["history"]
        assert "val_loss" in data["history"]
        assert len(data["history"]["loss"]) == 2

    # 4. Check CSV content
    df_history = pd.read_csv(history_csv)
    assert len(df_history) == 2
    assert "epoch" in df_history.columns
    assert "loss" in df_history.columns
    assert "val_loss" in df_history.columns


def test_train_baseline_missing_dataset_raises_error(tmp_path: Path):
    """Verify training on missing directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        train_baseline_model(
            data_source=tmp_path / "non_existent_folder",
            epochs=1,
            verbose=0,
        )
