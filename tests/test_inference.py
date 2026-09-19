"""Unit tests for model loading and inference contracts."""

from pathlib import Path
import numpy as np
import pytest

from src.config import DEFAULT_MODEL_PATH
from src.inference.predictor import load_detector_model, predict_tumor


def test_load_detector_model_missing_file_raises_error():
    """Verify load_detector_model raises FileNotFoundError on missing path."""
    non_existent = Path("models/does_not_exist_model.keras")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_detector_model(non_existent)


def test_predict_tumor_with_mock_model():
    """Verify predict_tumor contract with a mock model."""

    class MockKerasModel:
        def predict(self, x, verbose=0):
            # Returns a batch probability array of shape (1, 1)
            return np.array([[0.85]], dtype=np.float32)

    mock_model = MockKerasModel()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    prob, label = predict_tumor(mock_model, dummy_image, threshold=0.5)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert prob == pytest.approx(0.85)
    assert label == "Tumor"


def test_predict_tumor_negative_class():
    """Verify predict_tumor correctly labels probabilities below threshold."""

    class MockKerasModel:
        def predict(self, x, verbose=0):
            return np.array([[0.15]], dtype=np.float32)

    mock_model = MockKerasModel()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    prob, label = predict_tumor(mock_model, dummy_image, threshold=0.5)

    assert prob == pytest.approx(0.15)
    assert label == "No Tumor"
