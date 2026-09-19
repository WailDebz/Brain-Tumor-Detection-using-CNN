"""Unit tests for tf.data dataset loading pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data.loader import build_dataset_splits_tf, build_tf_dataset


@pytest.fixture
def synthetic_images_and_manifest(tmp_path: Path) -> pd.DataFrame:
    """Create a temporary directory with 8 synthetic images and a manifest DataFrame."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    for i in range(8):
        file_path = img_dir / f"scan_{i}.png"
        img = Image.new("RGB", (64, 64), color=(i * 30, i * 20, 100))
        img.save(file_path)

        records.append(
            {
                "filepath": str(file_path.resolve()),
                "filename": file_path.name,
                "class_name": "no" if i < 4 else "yes",
                "label": 0 if i < 4 else 1,
                "file_hash": f"mock_hash_{i}",
            }
        )

    return pd.DataFrame(records)


def test_build_tf_dataset_training(synthetic_images_and_manifest: pd.DataFrame):
    """Verify training dataset yields expected batch shapes and data types."""
    dataset = build_tf_dataset(
        manifest=synthetic_images_and_manifest,
        is_training=True,
        batch_size=4,
        target_size=(90, 90),
        shuffle_buffer=8,
        seed=42,
    )

    batch_count = 0
    for images, labels in dataset:
        batch_count += 1
        assert images.shape == (4, 90, 90, 1)
        assert labels.shape == (4,)
        assert images.dtype.name == "float32"
        assert labels.dtype.name == "float32"

    assert batch_count == 2  # 8 samples / batch_size 4 = 2 batches


def test_build_tf_dataset_validation(synthetic_images_and_manifest: pd.DataFrame):
    """Verify validation dataset yields unaugmented, deterministic batches."""
    dataset = build_tf_dataset(
        manifest=synthetic_images_and_manifest,
        is_training=False,
        batch_size=4,
        target_size=(90, 90),
    )

    batch_count = 0
    for images, labels in dataset:
        batch_count += 1
        assert images.shape == (4, 90, 90, 1)
        assert labels.shape == (4,)

    assert batch_count == 2


def test_build_dataset_splits_tf(synthetic_images_and_manifest: pd.DataFrame):
    """Verify build_dataset_splits_tf builds all three partitions."""
    train_df = synthetic_images_and_manifest.iloc[:4]
    val_df = synthetic_images_and_manifest.iloc[4:6]
    test_df = synthetic_images_and_manifest.iloc[6:]

    train_ds, val_ds, test_ds = build_dataset_splits_tf(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=2,
        target_size=(90, 90),
    )

    # Check train dataset has 2 batches (4 samples / 2)
    train_batches = list(train_ds)
    assert len(train_batches) == 2

    # Check val dataset has 1 batch (2 samples / 2)
    val_batches = list(val_ds)
    assert len(val_batches) == 1

    # Check test dataset has 1 batch (2 samples / 2)
    test_batches = list(test_ds)
    assert len(test_batches) == 1
