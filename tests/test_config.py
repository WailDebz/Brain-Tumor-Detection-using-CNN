"""Unit tests for project configuration and path resolution."""

from pathlib import Path
from src.config import (
    APP_DIR,
    CLASS_LABELS,
    DECISION_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    DOCS_DIR,
    MODELS_DIR,
    NUM_CHANNELS,
    PROJECT_ROOT,
)


def test_project_root_exists():
    """Verify PROJECT_ROOT resolves to an existing directory."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_expected_directories_exist():
    """Verify that expected top-level directories are resolved."""
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "app").is_dir()
    assert (PROJECT_ROOT / "models").is_dir()
    assert (PROJECT_ROOT / "docs").is_dir()
    assert (PROJECT_ROOT / "notebooks" / "archive").is_dir()


def test_default_model_path_resolution():
    """Verify default model path points inside MODELS_DIR with .keras extension."""
    assert DEFAULT_MODEL_PATH.parent == MODELS_DIR
    assert DEFAULT_MODEL_PATH.suffix == ".keras"
    assert DEFAULT_MODEL_PATH.exists(), f"Expected {DEFAULT_MODEL_PATH} to exist on disk."


def test_default_constants():
    """Verify defaults match the baseline model specifications."""
    assert DEFAULT_IMAGE_SIZE == (90, 90)
    assert NUM_CHANNELS == 1
    assert CLASS_LABELS == ["No Tumor", "Tumor"]
    assert 0.0 < DECISION_THRESHOLD < 1.0
