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

# Default model artifact path
DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "brain_tumor_detector.keras"

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

# DataLoader defaults
DEFAULT_BATCH_SIZE: Final[int] = 32

