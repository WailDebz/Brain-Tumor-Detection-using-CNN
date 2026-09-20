"""Project-wide path configurations and default constants."""

from pathlib import Path
from typing import Dict, Final, List, Set, Tuple

# Project root resolution (Brain-Tumor-Detection-using-CNN/)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Directory paths
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
APP_DIR: Final[Path] = PROJECT_ROOT / "app"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
REPORTS_METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"

# Model artifact paths
ORIGINAL_LEAKED_MODEL_PATH: Final[Path] = MODELS_DIR / "brain_tumor_detector.keras"
BASELINE_MODEL_PATH: Final[Path] = MODELS_DIR / "brain_tumor_cnn_baseline.keras"
TRANSFER_MODEL_PATH: Final[Path] = MODELS_DIR / "brain_tumor_mobilenetv2_frozen.keras"
DEFAULT_MODEL_PATH: Final[Path] = ORIGINAL_LEAKED_MODEL_PATH  # Maintained for app until new model evaluated

# Image processing specifications
DEFAULT_IMAGE_SIZE: Final[Tuple[int, int]] = (90, 90)
NUM_CHANNELS: Final[int] = 1
SUPPORTED_IMAGE_EXTENSIONS: Final[Set[str]] = {".jpg", ".jpeg", ".png"}

# Classification classes and label mappings
CLASS_LABELS: Final[List[str]] = ["No Tumor", "Tumor"]
CLASS_TO_LABEL: Final[Dict[str, int]] = {"no": 0, "yes": 1}
LABEL_TO_CLASS: Final[Dict[int, str]] = {0: "No Tumor", 1: "Tumor"}
DECISION_THRESHOLD: Final[float] = 0.5

# Reproducibility & Splitting configuration
RANDOM_SEED: Final[int] = 42

# Split ratios: (train, val, test)
# Rationale: For small medical datasets (~253 images), 70% train (approx. 177 images)
# leaves 15% validation (approx. 38 images) and 15% test (approx. 38 images), ensuring
# adequate unseen sample volume for statistically meaningful evaluation metrics.
DEFAULT_SPLIT_RATIOS: Final[Tuple[float, float, float]] = (0.70, 0.15, 0.15)

# DataLoader & Training defaults
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_EPOCHS: Final[int] = 30
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_L2_REGULARIZATION: Final[float] = 1e-4
DEFAULT_DROPOUT_RATE: Final[float] = 0.3

# Training Callbacks configuration
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 8
DEFAULT_REDUCE_LR_PATIENCE: Final[int] = 4
DEFAULT_REDUCE_LR_FACTOR: Final[float] = 0.5
DEFAULT_MIN_LR: Final[float] = 1e-6

# Default output reporting paths - Baseline CNN
DEFAULT_HISTORY_JSON_PATH: Final[Path] = REPORTS_METRICS_DIR / "training_history_baseline.json"
DEFAULT_HISTORY_CSV_PATH: Final[Path] = REPORTS_METRICS_DIR / "training_history_baseline.csv"
DEFAULT_TRAINING_CURVES_PATH: Final[Path] = REPORTS_FIGURES_DIR / "training_curves_baseline.png"

# Default output reporting paths - Transfer Learning (MobileNetV2 Frozen)
DEFAULT_TRANSFER_HISTORY_JSON_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "training_history_transfer_mobilenetv2_frozen.json"
)
DEFAULT_TRANSFER_HISTORY_CSV_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "training_history_transfer_mobilenetv2_frozen.csv"
)
DEFAULT_TRANSFER_TRAINING_CURVES_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "training_curves_transfer_mobilenetv2_frozen.png"
)

# Default evaluation paths
DEFAULT_TEST_MANIFEST_PATH: Final[Path] = PROCESSED_DATA_DIR / "test_manifest.csv"
DEFAULT_VAL_MANIFEST_PATH: Final[Path] = PROCESSED_DATA_DIR / "val_manifest.csv"
DEFAULT_TEST_METRICS_JSON_PATH: Final[Path] = REPORTS_METRICS_DIR / "test_metrics_baseline.json"
DEFAULT_TEST_METRICS_CSV_PATH: Final[Path] = REPORTS_METRICS_DIR / "test_metrics_baseline.csv"
DEFAULT_TRANSFER_TEST_METRICS_JSON_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "test_metrics_transfer_mobilenetv2_frozen.json"
)
DEFAULT_TRANSFER_TEST_METRICS_CSV_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "test_metrics_transfer_mobilenetv2_frozen.csv"
)
DEFAULT_TRANSFER_VAL_TUNED_JSON_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "test_metrics_transfer_val_tuned.json"
)
DEFAULT_TRANSFER_VAL_TUNED_CSV_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "test_metrics_transfer_val_tuned.csv"
)
# Default evaluation paths - Baseline CNN
DEFAULT_CONFUSION_MATRIX_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "confusion_matrix_test_brain_tumor_cnn_baseline.png"
)
DEFAULT_ROC_CURVE_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "roc_curve_test_brain_tumor_cnn_baseline.png"
)
DEFAULT_PR_CURVE_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "pr_curve_test_brain_tumor_cnn_baseline.png"
)

# Default evaluation paths - Transfer Learning (MobileNetV2 Frozen)
DEFAULT_TRANSFER_CONFUSION_MATRIX_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "confusion_matrix_test_brain_tumor_mobilenetv2_frozen.png"
)
DEFAULT_TRANSFER_ROC_CURVE_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "roc_curve_test_brain_tumor_mobilenetv2_frozen.png"
)
DEFAULT_TRANSFER_PR_CURVE_PATH: Final[Path] = (
    REPORTS_FIGURES_DIR / "pr_curve_test_brain_tumor_mobilenetv2_frozen.png"
)

# Multi-Model Comparison paths
DEFAULT_MODEL_COMPARISON_JSON_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "model_comparison.json"
)
DEFAULT_MODEL_COMPARISON_CSV_PATH: Final[Path] = (
    REPORTS_METRICS_DIR / "model_comparison.csv"
)



