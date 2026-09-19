"""Evaluation engine for held-out test benchmarking and validation threshold tuning."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    DECISION_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PR_CURVE_PATH,
    DEFAULT_ROC_CURVE_PATH,
    DEFAULT_TEST_MANIFEST_PATH,
    DEFAULT_TEST_METRICS_CSV_PATH,
    DEFAULT_TEST_METRICS_JSON_PATH,
    REPORTS_FIGURES_DIR,
)
from src.data.ingestion import load_manifest
from src.data.loader import build_tf_dataset
from src.evaluation.metrics import (
    EvaluationReport,
    apply_decision_threshold,
    compute_evaluation_report,
    find_optimal_threshold_on_validation,
)
from src.evaluation.visualizer import generate_all_evaluation_figures


def evaluate_model_on_manifest(
    model: Any,
    manifest: Union[pd.DataFrame, str, Path] = DEFAULT_TEST_MANIFEST_PATH,
    threshold: float = DECISION_THRESHOLD,
    threshold_strategy: str = "fixed_default_0.5",
    model_name: str = "brain_tumor_baseline_cnn",
    split_name: str = "test",
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    metrics_json_path: Optional[Union[str, Path]] = DEFAULT_TEST_METRICS_JSON_PATH,
    metrics_csv_path: Optional[Union[str, Path]] = DEFAULT_TEST_METRICS_CSV_PATH,
    figures_dir: Optional[Union[str, Path]] = REPORTS_FIGURES_DIR,
    generate_plots: bool = True,
    verbose: int = 1,
) -> Tuple[EvaluationReport, Dict[str, Path], pd.DataFrame]:
    """Execute evaluation of a trained model on a dataset partition manifest.

    EVALUATION PRINCIPLES:
    1. Deterministic Execution: Augmentation is strictly disabled; preprocessing is uniform.
    2. Data Isolation: Evaluates strictly on the provided manifest without modifying model weights.
    3. Comprehensive Reporting: Outputs structured EvaluationReport, metric CSV/JSON, and figures.

    Args:
        model: Loaded tf.keras.Model OR Path/string pointing to a saved .keras model.
        manifest: DataFrame or Path to manifest CSV containing 'filepath' and 'label'.
        threshold: Decision threshold for positive tumor class.
        threshold_strategy: Metadata description of how the threshold was determined.
        model_name: Model identifier name.
        split_name: Name of evaluated split ('test', 'val', 'train').
        batch_size: Batch size for forward pass.
        target_size: Input image dimensions (width, height).
        metrics_json_path: Optional destination path for JSON metric report.
        metrics_csv_path: Optional destination path for CSV metric report.
        figures_dir: Optional destination directory for diagnostic plots.
        generate_plots: Whether to generate and save confusion matrix, ROC, and PR plots.
        verbose: Verbosity level (0: silent, 1: print report summary).

    Returns:
        Tuple of (EvaluationReport, dict_of_figure_paths, predictions_dataframe).

    Raises:
        FileNotFoundError: If model file or manifest file is not found.
        ValueError: If manifest is empty or missing required columns.
    """
    import tensorflow as tf

    # Step 1: Load model if a path is provided
    if isinstance(model, (str, Path)):
        model_path = Path(model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path.resolve()}")
        loaded_model = tf.keras.models.load_model(str(model_path))
    else:
        loaded_model = model

    # Step 2: Load and validate manifest
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Evaluation manifest not found at: {manifest_path.resolve()}")
        df_eval = load_manifest(manifest_path)
    else:
        df_eval = manifest.copy()

    if len(df_eval) == 0:
        raise ValueError("Evaluation manifest contains 0 samples. Cannot evaluate.")

    if "filepath" not in df_eval.columns or "label" not in df_eval.columns:
        raise ValueError(f"Manifest missing required columns ('filepath', 'label'): {list(df_eval.columns)}")

    # Check for duplicate sample paths
    if df_eval["filepath"].duplicated().any():
        dup_count = df_eval["filepath"].duplicated().sum()
        raise ValueError(f"Evaluation manifest contains {dup_count} duplicate file paths.")

    # Step 3: Build deterministic, unaugmented tf.data.Dataset
    eval_ds = build_tf_dataset(
        manifest=df_eval,
        is_training=False,
        batch_size=batch_size,
        target_size=target_size,
        enable_augmentation=False,
    )

    # Step 4: Generate continuous probability predictions
    raw_preds = loaded_model.predict(eval_ds, verbose=0)
    y_prob = raw_preds.ravel() if raw_preds.ndim == 2 else raw_preds
    y_prob = np.asarray(y_prob, dtype=np.float32)

    y_true = np.asarray(df_eval["label"].values, dtype=int)
    y_pred = apply_decision_threshold(y_prob, threshold=threshold)

    # Step 5: Compute full evaluation metrics report
    report = compute_evaluation_report(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        model_name=model_name,
        dataset_split=split_name,
    )

    # Step 6: Save serialized reports
    if metrics_json_path is not None:
        report.save_json(metrics_json_path)

    if metrics_csv_path is not None:
        report.save_csv(metrics_csv_path)

    # Step 7: Generate and save diagnostic figures
    figure_paths: Dict[str, Path] = {}
    if generate_plots and figures_dir is not None:
        figure_paths = generate_all_evaluation_figures(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            figures_dir=figures_dir,
            model_name=model_name,
            split_name=split_name.capitalize(),
        )

    # Step 8: Build detailed prediction breakdown DataFrame
    predictions_df = df_eval.copy()
    predictions_df["predicted_probability"] = y_prob
    predictions_df["predicted_binary"] = y_pred
    predictions_df["predicted_label"] = np.where(y_pred == 1, "Tumor", "No Tumor")
    predictions_df["is_correct"] = predictions_df["label"] == predictions_df["predicted_binary"]

    if verbose > 0:
        print(report.summary())
        if figure_paths:
            print("Generated Evaluation Figures:")
            for fig_name, fig_path in figure_paths.items():
                print(f"  - {fig_name:20s}: {fig_path.resolve()}")

    return report, figure_paths, predictions_df


def evaluate_with_validation_threshold_tuning(
    model: Any,
    val_manifest: Union[pd.DataFrame, str, Path],
    test_manifest: Union[pd.DataFrame, str, Path],
    metric_for_threshold: str = "youden_j",
    model_name: str = "brain_tumor_baseline_cnn",
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    metrics_json_path: Optional[Union[str, Path]] = DEFAULT_TEST_METRICS_JSON_PATH,
    metrics_csv_path: Optional[Union[str, Path]] = DEFAULT_TEST_METRICS_CSV_PATH,
    figures_dir: Optional[Union[str, Path]] = REPORTS_FIGURES_DIR,
    verbose: int = 1,
) -> Tuple[EvaluationReport, Dict[str, Path], pd.DataFrame, float]:
    """Tune decision threshold on VALIDATION set, freeze it, and evaluate on TEST set.

    METHODOLOGICAL GUARANTEE:
    - Step 1: Run inference on Validation set.
    - Step 2: Select optimal threshold maximizing the selected validation criterion (e.g., Youden's J).
    - Step 3: Freeze threshold.
    - Step 4: Apply frozen threshold ONCE to the held-out Test set.
    - Test labels are NEVER exposed during threshold optimization.

    Args:
        model: Loaded model or model file path.
        val_manifest: Validation set manifest.
        test_manifest: Test set manifest.
        metric_for_threshold: Tuning criterion ('youden_j', 'f1', 'sensitivity_at_high_specificity').
        model_name: Model identifier.
        batch_size: Batch size.
        target_size: Target image dimensions.
        metrics_json_path: Destination for metrics JSON.
        metrics_csv_path: Destination for metrics CSV.
        figures_dir: Destination for plots.
        verbose: Verbosity level.

    Returns:
        Tuple of (EvaluationReport, figure_paths, test_predictions_df, chosen_threshold).
    """
    import tensorflow as tf

    # Load model if path
    if isinstance(model, (str, Path)):
        loaded_model = tf.keras.models.load_model(str(model))
    else:
        loaded_model = model

    # 1. Validation inference for threshold selection
    val_df = load_manifest(val_manifest) if isinstance(val_manifest, (str, Path)) else val_manifest.copy()
    val_ds = build_tf_dataset(
        manifest=val_df,
        is_training=False,
        batch_size=batch_size,
        target_size=target_size,
        enable_augmentation=False,
    )
    val_preds = loaded_model.predict(val_ds, verbose=0).ravel()
    val_true = val_df["label"].values

    optimal_threshold, best_val_score = find_optimal_threshold_on_validation(
        y_true_val=val_true,
        y_prob_val=val_preds,
        metric=metric_for_threshold,
    )

    if verbose > 0:
        print("\n" + "-" * 64)
        print(f"VALIDATION-BASED THRESHOLD SELECTION ({metric_for_threshold.upper()}):")
        print(f"  Selected Optimal Threshold : {optimal_threshold:.4f}")
        print(f"  Validation Score Achieved  : {best_val_score:.4f}")
        print("  -> Applying frozen threshold to held-out test partition...")
        print("-" * 64 + "\n")

    # 2. Evaluate held-out test set using frozen validation-tuned threshold
    report, fig_paths, pred_df = evaluate_model_on_manifest(
        model=loaded_model,
        manifest=test_manifest,
        threshold=optimal_threshold,
        threshold_strategy=f"validation_tuned_{metric_for_threshold}",
        model_name=model_name,
        split_name="test",
        batch_size=batch_size,
        target_size=target_size,
        metrics_json_path=metrics_json_path,
        metrics_csv_path=metrics_csv_path,
        figures_dir=figures_dir,
        generate_plots=True,
        verbose=verbose,
    )

    return report, fig_paths, pred_df, optimal_threshold
