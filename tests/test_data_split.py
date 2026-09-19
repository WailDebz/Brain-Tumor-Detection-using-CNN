"""Unit tests for leak-free stratified splitting and safety verification."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.split import (
    create_stratified_splits,
    save_splits,
    verify_split_leak_safety,
)


@pytest.fixture
def synthetic_manifest_df() -> pd.DataFrame:
    """Create a 100-sample synthetic manifest with 40 'no' (0) and 60 'yes' (1)."""
    records = []
    for i in range(100):
        label_int = 0 if i < 40 else 1
        class_name = "no" if label_int == 0 else "yes"
        records.append(
            {
                "filepath": f"/mock/path/image_{i:03d}.jpg",
                "filename": f"image_{i:03d}.jpg",
                "class_name": class_name,
                "label_name": "No Tumor" if label_int == 0 else "Tumor",
                "label": label_int,
                "file_hash": f"hash_{i:03d}_{class_name}",
                "width": 100,
                "height": 100,
                "channels_mode": "RGB",
            }
        )
    return pd.DataFrame(records)


def test_stratified_split_proportions(synthetic_manifest_df: pd.DataFrame):
    """Verify split ratios generate exact expected partition sizes."""
    train_df, val_df, test_df, report = create_stratified_splits(
        synthetic_manifest_df,
        split_ratios=(0.70, 0.15, 0.15),
        seed=42,
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert report.total_count == 100
    assert report.is_leak_free is True


def test_stratification_preserves_class_balance(synthetic_manifest_df: pd.DataFrame):
    """Verify class proportions (~40% no, ~60% yes) are preserved across all splits."""
    train_df, val_df, test_df, _ = create_stratified_splits(
        synthetic_manifest_df,
        split_ratios=(0.70, 0.15, 0.15),
        seed=42,
    )

    # Train: 40% of 70 = 28 'no', 42 'yes'
    assert (train_df["label"] == 0).sum() == 28
    assert (train_df["label"] == 1).sum() == 42

    # Val: 40% of 15 = 6 'no', 9 'yes'
    assert (val_df["label"] == 0).sum() == 6
    assert (val_df["label"] == 1).sum() == 9

    # Test: 40% of 15 = 6 'no', 9 'yes'
    assert (test_df["label"] == 0).sum() == 6
    assert (test_df["label"] == 1).sum() == 9


def test_split_reproducibility(synthetic_manifest_df: pd.DataFrame):
    """Verify identical random seeds produce identical split assignments."""
    train1, val1, test1, _ = create_stratified_splits(synthetic_manifest_df, seed=42)
    train2, val2, test2, _ = create_stratified_splits(synthetic_manifest_df, seed=42)

    assert list(train1["filepath"]) == list(train2["filepath"])
    assert list(val1["filepath"]) == list(val2["filepath"])
    assert list(test1["filepath"]) == list(test2["filepath"])


def test_zero_leakage_safety_verification(synthetic_manifest_df: pd.DataFrame):
    """Verify zero overlap between train, val, and test subsets."""
    train_df, val_df, test_df, report = create_stratified_splits(synthetic_manifest_df, seed=42)

    is_safe, p_overlaps, h_overlaps = verify_split_leak_safety(train_df, val_df, test_df)
    assert is_safe is True
    assert p_overlaps == 0
    assert h_overlaps == 0
    assert report.filepath_overlap_count == 0


def test_leakage_detector_catches_contamination(synthetic_manifest_df: pd.DataFrame):
    """Verify verify_split_leak_safety raises AssertionError when contaminated data is passed."""
    train_df, val_df, test_df, _ = create_stratified_splits(synthetic_manifest_df, seed=42)

    # Intentionally corrupt test_df by injecting a row from train_df
    contaminated_test = pd.concat([test_df, train_df.iloc[[0]]], ignore_index=True)

    with pytest.raises(AssertionError, match="CRITICAL DATA LEAKAGE DETECTED"):
        verify_split_leak_safety(train_df, val_df, contaminated_test)


def test_invalid_split_ratios_raise_error(synthetic_manifest_df: pd.DataFrame):
    """Verify ratios not summing to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        create_stratified_splits(synthetic_manifest_df, split_ratios=(0.6, 0.2, 0.1))


def test_save_splits_to_disk(synthetic_manifest_df: pd.DataFrame, tmp_path: Path):
    """Verify saving splits generates valid CSV files."""
    train_df, val_df, test_df, _ = create_stratified_splits(synthetic_manifest_df, seed=42)
    out_dir = tmp_path / "splits"

    paths = save_splits(train_df, val_df, test_df, output_dir=out_dir)

    for split_key in ["train", "val", "test", "combined"]:
        assert paths[split_key].exists()
        loaded = pd.read_csv(paths[split_key])
        assert len(loaded) > 0
