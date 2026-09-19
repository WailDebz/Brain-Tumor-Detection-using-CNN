"""Data ingestion, dataset loading, and leak-free splitting modules."""

from src.data.augment import build_augmentation_layer
from src.data.ingestion import (
    IngestionReport,
    compute_file_hash,
    load_manifest,
    save_manifest,
    scan_and_ingest_data,
)
from src.data.loader import build_dataset_splits_tf, build_tf_dataset
from src.data.preprocessing import (
    load_and_preprocess_image,
    preprocess_for_inference,
    preprocess_image_array,
)
from src.data.split import (
    SplitReport,
    create_stratified_splits,
    save_splits,
    verify_split_leak_safety,
)

__all__ = [
    "IngestionReport",
    "SplitReport",
    "build_augmentation_layer",
    "build_dataset_splits_tf",
    "build_tf_dataset",
    "compute_file_hash",
    "create_stratified_splits",
    "load_and_preprocess_image",
    "load_manifest",
    "preprocess_for_inference",
    "preprocess_image_array",
    "save_manifest",
    "save_splits",
    "scan_and_ingest_data",
    "verify_split_leak_safety",
]

