"""Unit tests for data ingestion, validation, and duplicate detection."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data.ingestion import (
    compute_file_hash,
    load_manifest,
    normalize_class_name,
    save_manifest,
    scan_and_ingest_data,
    validate_image_file,
)


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic dataset directory with valid, corrupt, and duplicate images."""
    data_dir = tmp_path / "raw"
    no_dir = data_dir / "no"
    yes_dir = data_dir / "yes"

    no_dir.mkdir(parents=True)
    yes_dir.mkdir(parents=True)

    # Create 10 valid images for 'no' class
    for i in range(10):
        img_array = np.full((100, 100, 3), fill_value=i * 20, dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(no_dir / f"no_img_{i}.jpg")

    # Create 15 valid images for 'yes' class (mix of .png, .jpeg, and uppercase .JPG)
    for i in range(15):
        img_array = np.full((120, 120, 3), fill_value=(i + 1) * 15, dtype=np.uint8)
        img = Image.fromarray(img_array)
        ext = ".png" if i % 2 == 0 else ".JPG"
        img.save(yes_dir / f"yes_img_{i}{ext}")

    # Create 1 corrupt/empty file in 'no'
    corrupt_file = no_dir / "corrupt_image.jpg"
    corrupt_file.write_bytes(b"not a valid image content")

    # Create 1 exact duplicate in 'yes' (copy of yes_img_0.png)
    orig_content = (yes_dir / "yes_img_0.png").read_bytes()
    (yes_dir / "yes_img_0_duplicate.png").write_bytes(orig_content)

    return data_dir


def test_normalize_class_name():
    """Verify class folder normalization mappings."""
    assert normalize_class_name("no") == "no"
    assert normalize_class_name("No") == "no"
    assert normalize_class_name("no_tumor") == "no"
    assert normalize_class_name("yes") == "yes"
    assert normalize_class_name("Tumor") == "yes"
    assert normalize_class_name("unrelated_folder") is None


def test_compute_file_hash(tmp_path: Path):
    """Verify SHA-256 hash generation is deterministic and unique per content."""
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    f3 = tmp_path / "file3.txt"

    f1.write_bytes(b"data content A")
    f2.write_bytes(b"data content A")
    f3.write_bytes(b"data content B")

    hash1 = compute_file_hash(f1)
    hash2 = compute_file_hash(f2)
    hash3 = compute_file_hash(f3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_validate_image_file_valid_and_corrupt(tmp_path: Path):
    """Verify image validation differentiates valid vs. corrupt image files."""
    valid_file = tmp_path / "valid.png"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(valid_file)

    is_valid, err, meta = validate_image_file(valid_file)
    assert is_valid is True
    assert err is None
    assert meta == (64, 64, "RGB")

    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"bad bytes")

    is_valid_c, err_c, meta_c = validate_image_file(corrupt_file)
    assert is_valid_c is False
    assert err_c is not None
    assert meta_c is None


def test_scan_and_ingest_data(synthetic_dataset_dir: Path):
    """Verify scan_and_ingest_data creates valid manifest and flags duplicates/corrupt files."""
    df, report = scan_and_ingest_data(synthetic_dataset_dir)

    # 10 valid 'no' + 15 valid 'yes' = 25 valid unique samples
    assert len(df) == 25
    assert report.valid_samples == 25
    assert report.class_counts["no"] == 10
    assert report.class_counts["yes"] == 15
    assert len(report.corrupt_files) == 1
    assert len(report.duplicate_files) == 1

    # Check required columns
    expected_cols = {"filepath", "filename", "class_name", "label", "file_hash", "width", "height"}
    assert expected_cols.issubset(set(df.columns))

    # Verify paths are portable and formatted with forward slashes
    for p in df["filepath"]:
        assert "\\" not in p

    for rel in df["relative_path"]:
        assert "\\" not in rel
        assert not rel.startswith("C:")


def test_scan_missing_dir_raises_error(tmp_path: Path):
    """Verify scanning a non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        scan_and_ingest_data(tmp_path / "non_existent_folder")


def test_scan_missing_class_raises_error(tmp_path: Path):
    """Verify scanning a dataset with only one class raises ValueError."""
    single_class_dir = tmp_path / "single" / "no"
    single_class_dir.mkdir(parents=True)
    img = Image.new("RGB", (50, 50))
    img.save(single_class_dir / "img1.jpg")

    with pytest.raises(ValueError, match="missing required classes"):
        scan_and_ingest_data(tmp_path / "single")


def test_save_and_load_manifest(synthetic_dataset_dir: Path, tmp_path: Path):
    """Verify manifest serialization to CSV and deserialization."""
    df, _ = scan_and_ingest_data(synthetic_dataset_dir)
    manifest_csv = tmp_path / "output" / "manifest.csv"

    saved_path = save_manifest(df, manifest_csv)
    assert saved_path.exists()

    loaded_df = load_manifest(manifest_csv)
    assert len(loaded_df) == len(df)
    assert list(loaded_df["filepath"]) == list(df["filepath"])
