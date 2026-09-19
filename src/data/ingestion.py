"""Data ingestion, validation, duplicate detection, and manifest generation."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from PIL import Image

from src.config import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
)


@dataclass
class IngestionReport:
    """Summary of data ingestion and validation results."""

    total_scanned: int = 0
    valid_samples: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_proportions: Dict[str, float] = field(default_factory=dict)
    corrupt_files: List[Tuple[str, str]] = field(default_factory=list)
    duplicate_files: List[Tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Format a human-readable ingestion summary."""
        lines = [
            "=" * 60,
            "DATA INGESTION & INTEGRITY REPORT",
            "=" * 60,
            f"Total files scanned: {self.total_scanned}",
            f"Valid image samples: {self.valid_samples}",
            f"Corrupt / unreadable: {len(self.corrupt_files)}",
            f"Duplicate files (SHA-256): {len(self.duplicate_files)}",
            "-" * 60,
            "Class Distribution:",
        ]
        for class_name, count in self.class_counts.items():
            prop = self.class_proportions.get(class_name, 0.0) * 100
            lines.append(f"  - {class_name:10s}: {count:5d} ({prop:5.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file for exact duplicate detection.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_class_name(folder_name: str) -> Optional[str]:
    """Map directory name to canonical class name ('no' or 'yes').

    Args:
        folder_name: Subdirectory name in raw data folder.

    Returns:
        Canonical class name ('no' or 'yes') or None if unrecognized.
    """
    clean = folder_name.strip().lower()
    if clean in {"no", "no_tumor", "notumor", "negative", "0"}:
        return "no"
    if clean in {"yes", "tumor", "positive", "1"}:
        return "yes"
    return None


def validate_image_file(file_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int, str]]]:
    """Check whether an image file is readable and uncorrupted.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple of (is_valid, error_message, (width, height, mode)).
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open to read dimensions/mode as verify() can leave file pointer spent
        with Image.open(file_path) as img:
            return True, None, (img.width, img.height, img.mode)
    except Exception as exc:
        return False, str(exc), None


def scan_and_ingest_data(
    data_dir: Optional[Union[str, Path]] = None,
    supported_extensions: Set[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> Tuple[pd.DataFrame, IngestionReport]:
    """Scan raw data directory, validate images, detect exact duplicates, and create manifest.

    Args:
        data_dir: Path to directory containing class subdirectories ('no', 'yes').
            Defaults to RAW_DATA_DIR.
        supported_extensions: Set of supported lowercase file extensions.

    Returns:
        Tuple of (manifest_dataframe, ingestion_report).

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no valid images are found or if class balance cannot be formed.
    """
    target_dir = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {target_dir.resolve()}.\n"
            "Please ensure raw dataset is placed in 'data/raw/' with 'no/' and 'yes/' subfolders."
        )

    records: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    report = IngestionReport()

    # Find subdirectories
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise ValueError(
            f"No subdirectories found in {target_dir.resolve()}. Expected class folders like 'no/' and 'yes/'."
        )

    for subdir in sorted(subdirs):
        canonical_class = normalize_class_name(subdir.name)
        if canonical_class is None:
            continue

        label_int = CLASS_TO_LABEL[canonical_class]
        label_display = LABEL_TO_CLASS[label_int]

        for file_path in sorted(subdir.iterdir()):
            if not file_path.is_file():
                continue

            report.total_scanned += 1
            ext = file_path.suffix.lower()

            if ext not in supported_extensions:
                continue

            # Validate image file readability
            is_valid, err_msg, meta = validate_image_file(file_path)
            if not is_valid:
                report.corrupt_files.append((str(file_path), err_msg or "Unknown error"))
                continue

            # Check duplicate file hash
            file_hash = compute_file_hash(file_path)
            if file_hash in seen_hashes:
                original_file = seen_hashes[file_hash]
                report.duplicate_files.append((str(file_path), original_file))
                # Skip exact duplicates to prevent split contamination
                continue
            seen_hashes[file_hash] = str(file_path)

            # Store repository-relative portable POSIX path if inside project, else POSIX absolute
            try:
                portable_filepath = file_path.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                portable_filepath = file_path.resolve().as_posix()

            width, height, mode = meta if meta is not None else (0, 0, "unknown")
            records.append(
                {
                    "filepath": portable_filepath,
                    "relative_path": file_path.relative_to(target_dir).as_posix(),
                    "filename": file_path.name,
                    "class_name": canonical_class,
                    "label_name": label_display,
                    "label": label_int,
                    "file_hash": file_hash,
                    "width": width,
                    "height": height,
                    "channels_mode": mode,
                }
            )

    if not records:
        raise ValueError(
            f"No valid images found in {target_dir.resolve()}. "
            f"Scanned {report.total_scanned} files across subdirectories."
        )

    df = pd.DataFrame(records)
    report.valid_samples = len(df)

    # Class distribution
    for c_name in CLASS_TO_LABEL:
        count = int((df["class_name"] == c_name).sum())
        report.class_counts[c_name] = count
        report.class_proportions[c_name] = count / len(df) if len(df) > 0 else 0.0

    # Ensure all expected binary classes are present
    missing_classes = [c for c, count in report.class_counts.items() if count == 0]
    if missing_classes:
        raise ValueError(
            f"Dataset is missing required classes: {missing_classes}. Found classes: {report.class_counts}"
        )

    return df, report


def save_manifest(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Save dataset manifest DataFrame to a CSV file.

    Args:
        df: Manifest DataFrame.
        output_path: Target CSV file path.

    Returns:
        Path of saved manifest CSV.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def load_manifest(manifest_path: Union[str, Path]) -> pd.DataFrame:
    """Load a dataset manifest from CSV and validate expected schema.

    Args:
        manifest_path: Path to manifest CSV.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path.resolve()}")

    df = pd.read_csv(path)
    required_cols = {"filepath", "label", "class_name", "file_hash"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path.name} is missing required columns: {missing}")

    return df
