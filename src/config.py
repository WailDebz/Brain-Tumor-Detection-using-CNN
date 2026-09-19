"""Project-wide path configurations and default constants."""

from pathlib import Path
from typing import Final, List, Tuple

# Project root resolution (Brain-Tumor-Detection-using-CNN/)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Directory paths
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
APP_DIR: Final[Path] = PROJECT_ROOT / "app"

# Default model artifact path
DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "brain_tumor_detector.keras"

# Image processing defaults
DEFAULT_IMAGE_SIZE: Final[Tuple[int, int]] = (90, 90)
NUM_CHANNELS: Final[int] = 1

# Classification labels mapping
CLASS_LABELS: Final[List[str]] = ["No Tumor", "Tumor"]
DECISION_THRESHOLD: Final[float] = 0.5
