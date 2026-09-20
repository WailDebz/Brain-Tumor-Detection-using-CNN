"""Training pipeline for the modernized baseline CNN on leak-free dataset splits."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNING_LEARNING_RATE,
    DEFAULT_HISTORY_CSV_PATH,
    DEFAULT_HISTORY_JSON_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_LR,
    DEFAULT_REDUCE_LR_FACTOR,
    DEFAULT_REDUCE_LR_PATIENCE,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TRAINING_CURVES_PATH,
    DEFAULT_UNFREEZE_FROM_LAYER,
    NUM_CHANNELS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TRANSFER_MODEL_PATH,
)
from src.data.ingestion import scan_and_ingest_data
from src.data.loader import build_tf_dataset
from src.data.split import SplitReport, create_stratified_splits, verify_split_leak_safety
from src.models.cnn import build_cnn_model
from src.models.transfer import build_transfer_model
from src.utils.reproducibility import set_random_seeds


@dataclass
class TrainingResult:
    """Structured container for training experiment outputs and metadata."""

    model_path: Path
    history_json_path: Path
    history_csv_path: Path
    plot_path: Path
    epochs_requested: int
    epochs_completed: int
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    history: Dict[str, List[float]] = field(default_factory=dict)
    split_report: Optional[SplitReport] = None

    def summary(self) -> str:
        """Format a concise human-readable training summary."""
        lines = [
            "=" * 60,
            "BASELINE CNN TRAINING EXPERIMENT SUMMARY",
            "=" * 60,
            f"Epochs Requested       : {self.epochs_requested}",
            f"Epochs Completed       : {self.epochs_completed}",
            f"Best Epoch (by Val Loss): {self.best_epoch}",
            f"Best Validation Loss   : {self.best_val_loss:.4f}",
            f"Best Validation Accuracy: {self.best_val_accuracy * 100:.2f}%",
            "-" * 60,
            "Generated Artifacts:",
            f"  - Model Saved to      : {self.model_path.resolve()}",
            f"  - History JSON        : {self.history_json_path.resolve()}",
            f"  - History CSV         : {self.history_csv_path.resolve()}",
            f"  - Training Curves Plot: {self.plot_path.resolve()}",
            "=" * 60,
        ]
        return "\n".join(lines)


def plot_and_save_training_curves(
    history_dict: Dict[str, List[float]],
    output_path: Path,
) -> Path:
    """Plot and save training vs. validation loss and accuracy curves.

    Args:
        history_dict: Dictionary containing 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        output_path: Target PNG file path.

    Returns:
        Path of saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for headless execution
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history_dict.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Loss Curve
    axes[0].plot(epochs_range, history_dict.get("loss", []), label="Train Loss", marker="o")
    axes[0].plot(epochs_range, history_dict.get("val_loss", []), label="Validation Loss", marker="s")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy Curve
    axes[1].plot(epochs_range, history_dict.get("accuracy", []), label="Train Accuracy", marker="o")
    axes[1].plot(epochs_range, history_dict.get("val_accuracy", []), label="Validation Accuracy", marker="s")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def train_baseline_model(
    data_source: Union[str, Path, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = RAW_DATA_DIR,
    model_type: str = "cnn",
    model_output_path: Path = BASELINE_MODEL_PATH,
    history_json_path: Path = DEFAULT_HISTORY_JSON_PATH,
    history_csv_path: Path = DEFAULT_HISTORY_CSV_PATH,
    plot_output_path: Path = DEFAULT_TRAINING_CURVES_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_reg: float = DEFAULT_L2_REGULARIZATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    reduce_lr_patience: int = DEFAULT_REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = DEFAULT_REDUCE_LR_FACTOR,
    min_lr: float = DEFAULT_MIN_LR,
    seed: int = RANDOM_SEED,
    verbose: int = 1,
) -> Tuple[Any, TrainingResult]:
    """Execute end-to-end training of the modernized baseline CNN.

    STRICT METHODOLOGICAL CONSTRAINTS:
    1. Dataset splitting is performed on original samples before any augmentation.
    2. Online data augmentation is enabled ONLY on the training dataset pipeline.
    3. Validation data is evaluated unaugmented to monitor generalization.
    4. The held-out test partition is NEVER passed to `model.fit()` and is reserved
       strictly for post-training clinical evaluation.

    Args:
        data_source: Path to raw dataset folder OR tuple of (train_df, val_df, test_df).
        model_output_path: Path to save the best model artifact.
        history_json_path: Path to save training history JSON.
        history_csv_path: Path to save training history CSV.
        plot_output_path: Path to save loss/accuracy curves.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial Adam learning rate.
        l2_reg: L2 regularization penalty.
        dropout_rate: Dropout rate.
        split_ratios: (train, val, test) proportions.
        early_stopping_patience: Early stopping patience epochs.
        reduce_lr_patience: ReduceLROnPlateau patience epochs.
        reduce_lr_factor: Multiplicative factor for LR reduction.
        min_lr: Lower bound on learning rate.
        seed: Random seed for reproducibility.
        verbose: Verbosity level (0, 1, or 2).

    Returns:
        Tuple of (trained_keras_model, TrainingResult).

    Raises:
        AssertionError: If data leakage is detected.
        FileNotFoundError: If raw dataset is missing.
    """
    import tensorflow as tf

    # Step 1: Set deterministic random seeds
    set_random_seeds(seed)

    # Step 2: Prepare or retrieve split manifests
    split_report: Optional[SplitReport] = None
    if isinstance(data_source, tuple) and len(data_source) == 3:
        train_df, val_df, test_df = data_source
        is_safe, p_ov, h_ov = verify_split_leak_safety(train_df, val_df, test_df)
        split_report = SplitReport(
            train_count=len(train_df),
            val_count=len(val_df),
            test_count=len(test_df),
            total_count=len(train_df) + len(val_df) + len(test_df),
            train_class_distribution=train_df["class_name"].value_counts().to_dict(),
            val_class_distribution=val_df["class_name"].value_counts().to_dict(),
            test_class_distribution=test_df["class_name"].value_counts().to_dict(),
            is_leak_free=is_safe,
            filepath_overlap_count=p_ov,
            hash_overlap_count=h_ov,
        )
    else:
        # Scan raw data directory and create stratified splits
        raw_path = Path(data_source)
        df, _ = scan_and_ingest_data(raw_path)
        train_df, val_df, test_df, split_report = create_stratified_splits(
            df=df,
            split_ratios=split_ratios,
            seed=seed,
        )

    # Log pre-training data audit
    if verbose > 0 and split_report is not None:
        print(split_report.summary())

    # Step 3: Build tf.data.Datasets
    # Training: Has online data augmentation enabled
    train_ds = build_tf_dataset(
        manifest=train_df,
        is_training=True,
        batch_size=batch_size,
        target_size=DEFAULT_IMAGE_SIZE,
        enable_augmentation=True,
        seed=seed,
    )

    # Validation: Unaugmented, deterministic
    val_ds = build_tf_dataset(
        manifest=val_df,
        is_training=False,
        batch_size=batch_size,
        target_size=DEFAULT_IMAGE_SIZE,
        enable_augmentation=False,
        seed=seed,
    )

    # Step 4: Build Model Architecture (Custom CNN, Frozen Transfer, or Fine-Tuned Transfer)
    input_shape = (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], NUM_CHANNELS)
    if model_type.lower() in {"transfer_mobilenetv2_finetuned", "transfer_finetuned", "finetuned"}:
        eff_lr = learning_rate if learning_rate != DEFAULT_LEARNING_RATE else DEFAULT_FINETUNING_LEARNING_RATE
        warm_start_weights = TRANSFER_MODEL_PATH if TRANSFER_MODEL_PATH.exists() else None
        model = build_transfer_model(
            input_shape=input_shape,
            learning_rate=eff_lr,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            fine_tune=True,
            unfreeze_from_layer=DEFAULT_UNFREEZE_FROM_LAYER,
            weights_path=warm_start_weights,
            name="brain_tumor_mobilenetv2_finetuned",
        )
    elif model_type.lower() in {"transfer", "transfer_mobilenetv2", "mobilenetv2", "transfer_frozen"}:
        model = build_transfer_model(
            input_shape=input_shape,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            fine_tune=False,
            name="brain_tumor_transfer_mobilenetv2_frozen",
        )
    else:
        model = build_cnn_model(
            input_shape=input_shape,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            name="brain_tumor_baseline_cnn",
        )

    if verbose > 0:
        model.summary()

    # Step 5: Configure Standard Keras Callbacks
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
            verbose=verbose,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_output_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=verbose,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min",
            verbose=verbose,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(history_csv_path),
            separator=",",
            append=False,
        ),
    ]

    # Step 6: Execute Training (Validation data only; Test data NEVER passed here)
    if verbose > 0:
        print(f"\n[TRAINING] Beginning model fit for up to {epochs} epochs...")

    fit_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Ensure best model is saved to designated path
    model.save(str(model_output_path))

    # Step 7: Parse history and compute key metrics
    history_dict: Dict[str, List[float]] = {
        k: [float(val) for val in v] for k, v in fit_history.history.items()
    }

    val_losses = history_dict.get("val_loss", [0.0])
    val_accuracies = history_dict.get("val_accuracy", [0.0])
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(val_losses[best_epoch_idx])
    best_val_accuracy = float(val_accuracies[best_epoch_idx])
    epochs_completed = len(val_losses)

    # Step 8: Save history to JSON
    history_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_json_path, "w") as f:
        json.dump(
            {
                "epochs_requested": epochs,
                "epochs_completed": epochs_completed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "history": history_dict,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "l2_reg": l2_reg,
                    "dropout_rate": dropout_rate,
                    "early_stopping_patience": early_stopping_patience,
                    "reduce_lr_patience": reduce_lr_patience,
                    "seed": seed,
                },
            },
            f,
            indent=2,
        )

    # Step 9: Generate & Save Training Curves Plot
    plot_and_save_training_curves(history_dict, plot_output_path)

    result = TrainingResult(
        model_path=model_output_path,
        history_json_path=history_json_path,
        history_csv_path=history_csv_path,
        plot_path=plot_output_path,
        epochs_requested=epochs,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        history=history_dict,
        split_report=split_report,
    )

    if verbose > 0:
        print(result.summary())

    return model, result
