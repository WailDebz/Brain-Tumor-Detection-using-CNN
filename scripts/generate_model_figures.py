"""Generate model-specific evaluation figures for both trained models."""

from pathlib import Path
import pandas as pd
import tensorflow as tf

from src.evaluation.evaluator import evaluate_model_on_manifest
from src.config import BASELINE_MODEL_PATH, TRANSFER_MODEL_PATH, DEFAULT_TEST_MANIFEST_PATH, REPORTS_FIGURES_DIR

# 1. Custom CNN Baseline
print("Generating figures for Custom CNN Baseline...")
evaluate_model_on_manifest(
    model=BASELINE_MODEL_PATH,
    manifest=DEFAULT_TEST_MANIFEST_PATH,
    threshold=0.5,
    threshold_strategy="fixed_default_0.5",
    model_name="brain_tumor_cnn_baseline",
    split_name="test",
    figures_dir=REPORTS_FIGURES_DIR,
    generate_plots=True,
    verbose=1,
)

# 2. MobileNetV2 Frozen
print("\nGenerating figures for MobileNetV2 Frozen...")
evaluate_model_on_manifest(
    model=TRANSFER_MODEL_PATH,
    manifest=DEFAULT_TEST_MANIFEST_PATH,
    threshold=0.5,
    threshold_strategy="fixed_default_0.5",
    model_name="brain_tumor_mobilenetv2_frozen",
    split_name="test",
    figures_dir=REPORTS_FIGURES_DIR,
    generate_plots=True,
    verbose=1,
)

print("\nModel-specific evaluation figures successfully generated.")
