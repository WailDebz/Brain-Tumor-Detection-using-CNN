"""Leak-free stratified dataset splitting and contamination verification."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_SPLIT_RATIOS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)


@dataclass
class SplitReport:
    """Detailed summary of dataset splits and verification checks."""

    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    total_count: int = 0
    train_class_distribution: Dict[str, int] = field(default_factory=dict)
    val_class_distribution: Dict[str, int] = field(default_factory=dict)
    test_class_distribution: Dict[str, int] = field(default_factory=dict)
    is_leak_free: bool = False
    filepath_overlap_count: int = 0
    hash_overlap_count: int = 0

    def summary(self) -> str:
        """Format a human-readable split summary."""
        lines = [
            "=" * 60,
            "DATASET STRATIFIED SPLIT & SAFETY AUDIT",
            "=" * 60,
            f"Total Original Samples : {self.total_count}",
            f"Train Set Size         : {self.train_count:4d} ({self.train_count / self.total_count * 100:5.1f}%)"
            if self.total_count
            else "",
            f"Validation Set Size    : {self.val_count:4d} ({self.val_count / self.total_count * 100:5.1f}%)"
            if self.total_count
            else "",
            f"Test Set Size          : {self.test_count:4d} ({self.test_count / self.total_count * 100:5.1f}%)"
            if self.total_count
            else "",
            "-" * 60,
            "Class Distribution Across Splits:",
            f"  Train : {self.train_class_distribution}",
            f"  Val   : {self.val_class_distribution}",
            f"  Test  : {self.test_class_distribution}",
            "-" * 60,
            "Leakage & Contamination Audit:",
            f"  Filepath Overlap Count : {self.filepath_overlap_count} (Expected: 0)",
            f"  File Hash Overlap Count: {self.hash_overlap_count} (Expected: 0)",
            f"  Leak-Free Status       : {'PASSED (Zero Overlap)' if self.is_leak_free else 'FAILED (Contamination Detected)'}",
            "=" * 60,
        ]
        return "\n".join(lines)


def verify_split_leak_safety(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[bool, int, int]:
    """Verify that there is zero sample overlap (by filepath or SHA-256 hash) across splits.

    Args:
        train_df: Training partition manifest.
        val_df: Validation partition manifest.
        test_df: Test partition manifest.

    Returns:
        Tuple of (is_safe, filepath_overlap_count, hash_overlap_count).

    Raises:
        AssertionError: If any data leakage or overlap is detected.
    """
    train_paths: Set[str] = set(train_df["filepath"])
    val_paths: Set[str] = set(val_df["filepath"])
    test_paths: Set[str] = set(test_df["filepath"])

    train_hashes: Set[str] = set(train_df["file_hash"])
    val_hashes: Set[str] = set(val_df["file_hash"])
    test_hashes: Set[str] = set(test_df["file_hash"])

    path_overlaps = (
        (train_paths & val_paths)
        | (train_paths & test_paths)
        | (val_paths & test_paths)
    )

    hash_overlaps = (
        (train_hashes & val_hashes)
        | (train_hashes & test_hashes)
        | (val_hashes & test_hashes)
    )

    is_safe = len(path_overlaps) == 0 and len(hash_overlaps) == 0
    if not is_safe:
        raise AssertionError(
            f"CRITICAL DATA LEAKAGE DETECTED across splits!\n"
            f"Filepath overlaps ({len(path_overlaps)}): {list(path_overlaps)[:5]}\n"
            f"Hash overlaps ({len(hash_overlaps)}): {list(hash_overlaps)[:5]}"
        )

    return is_safe, len(path_overlaps), len(hash_overlaps)


def create_stratified_splits(
    df: pd.DataFrame,
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split the dataset into Train, Validation, and Test sets with stratification.

    IMPORTANT:
    This operation runs exclusively on the ORIGINAL raw images prior to any augmentation,
    guaranteeing that synthetic or transformed variants of the same base scan cannot appear
    across multiple evaluation subsets.

    Args:
        df: Input manifest DataFrame containing 'filepath', 'label', 'class_name', 'file_hash'.
        split_ratios: (train_ratio, val_ratio, test_ratio) summing to 1.0.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df, split_report).

    Raises:
        ValueError: If split_ratios do not sum to 1.0 or if dataset is too small.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-4):
        raise ValueError(
            f"Split ratios must sum to 1.0. Received: {split_ratios} (Sum: {sum(split_ratios):.4f})"
        )

    if len(df) < 10:
        raise ValueError(f"Dataset too small to split meaningfully. Received {len(df)} samples.")

    # First split: Separate Train from (Validation + Test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )

    # Second split: Divide (Validation + Test) into relative Validation and Test portions
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_val_test_df,
        test_size=relative_test_ratio,
        stratify=temp_val_test_df["label"],
        random_state=seed,
        shuffle=True,
    )

    # Copy and assign split identifiers
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Reset indices
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Safety verification
    is_safe, p_overlaps, h_overlaps = verify_split_leak_safety(train_df, val_df, test_df)

    # Compile report
    report = SplitReport(
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        total_count=len(df),
        train_class_distribution=train_df["class_name"].value_counts().to_dict(),
        val_class_distribution=val_df["class_name"].value_counts().to_dict(),
        test_class_distribution=test_df["class_name"].value_counts().to_dict(),
        is_leak_free=is_safe,
        filepath_overlap_count=p_overlaps,
        hash_overlap_count=h_overlaps,
    )

    return train_df, val_df, test_df, report


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Save train, val, test, and combined split manifests to disk.

    Args:
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.
        output_dir: Output directory for CSV files. Defaults to PROCESSED_DATA_DIR.

    Returns:
        Dictionary mapping split names to saved CSV Paths.
    """
    target_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": target_dir / "train_manifest.csv",
        "val": target_dir / "val_manifest.csv",
        "test": target_dir / "test_manifest.csv",
        "combined": target_dir / "full_split_manifest.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(paths["combined"], index=False)

    return paths
