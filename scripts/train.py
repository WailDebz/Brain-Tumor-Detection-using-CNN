"""CLI entrypoint for training the modernized baseline CNN on a leak-free dataset."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNED_HISTORY_CSV_PATH,
    DEFAULT_FINETUNED_HISTORY_JSON_PATH,
    DEFAULT_FINETUNED_TRAINING_CURVES_PATH,
    DEFAULT_FINETUNING_LEARNING_RATE,
    DEFAULT_HISTORY_CSV_PATH,
    DEFAULT_HISTORY_JSON_PATH,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_REDUCE_LR_FACTOR,
    DEFAULT_REDUCE_LR_PATIENCE,
    DEFAULT_TRAINING_CURVES_PATH,
    DEFAULT_TRANSFER_HISTORY_CSV_PATH,
    DEFAULT_TRANSFER_HISTORY_JSON_PATH,
    DEFAULT_TRANSFER_TRAINING_CURVES_PATH,
    FINETUNED_TRANSFER_MODEL_PATH,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TRANSFER_MODEL_PATH,
)
from src.training.trainer import train_baseline_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for baseline model training."""
    parser = argparse.ArgumentParser(
        description="Train the modernized baseline CNN or transfer learning model on a leak-free brain MRI dataset."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn",
        choices=[
            "cnn",
            "transfer_mobilenetv2",
            "transfer",
            "transfer_mobilenetv2_finetuned",
            "transfer_finetuned",
            "finetuned",
        ],
        help="Architecture type: 'cnn' (custom baseline), 'transfer_mobilenetv2' (frozen MobileNetV2), or 'transfer_mobilenetv2_finetuned' (fine-tuned MobileNetV2). Default: cnn",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory containing raw class subfolders ('no', 'yes'). Default: {RAW_DATA_DIR}",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Path to save the best trained .keras model artifact.",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Path to save training history JSON.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="Path to save training history CSV.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Path to save training curves PNG.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Maximum training epochs. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial Adam learning rate. Default: 1e-3 (CNN / Frozen) or 1e-5 (Fine-tuned).",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=DEFAULT_L2_REGULARIZATION,
        help=f"L2 kernel regularization penalty. Default: {DEFAULT_L2_REGULARIZATION}",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help=f"Dropout rate in classification head. Default: {DEFAULT_DROPOUT_RATE}",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience epochs. Default: {DEFAULT_EARLY_STOPPING_PATIENCE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for deterministic initialization. Default: {RANDOM_SEED}",
    )
    return parser.parse_args()


def main() -> int:
    """Execute training CLI."""
    args = parse_args()

    # Determine default paths and learning rate based on model type
    is_finetuned = args.model_type.lower() in {
        "transfer_mobilenetv2_finetuned",
        "transfer_finetuned",
        "finetuned",
    }
    is_transfer_frozen = args.model_type.lower() in {
        "transfer",
        "transfer_mobilenetv2",
        "mobilenetv2",
        "transfer_frozen",
    }

    if is_finetuned:
        default_model_output = FINETUNED_TRANSFER_MODEL_PATH
        default_history_json = DEFAULT_FINETUNED_HISTORY_JSON_PATH
        default_history_csv = DEFAULT_FINETUNED_HISTORY_CSV_PATH
        default_plot_output = DEFAULT_FINETUNED_TRAINING_CURVES_PATH
        effective_lr = args.lr if args.lr is not None else DEFAULT_FINETUNING_LEARNING_RATE
    elif is_transfer_frozen:
        default_model_output = TRANSFER_MODEL_PATH
        default_history_json = DEFAULT_TRANSFER_HISTORY_JSON_PATH
        default_history_csv = DEFAULT_TRANSFER_HISTORY_CSV_PATH
        default_plot_output = DEFAULT_TRANSFER_TRAINING_CURVES_PATH
        effective_lr = args.lr if args.lr is not None else DEFAULT_LEARNING_RATE
    else:
        default_model_output = BASELINE_MODEL_PATH
        default_history_json = DEFAULT_HISTORY_JSON_PATH
        default_history_csv = DEFAULT_HISTORY_CSV_PATH
        default_plot_output = DEFAULT_TRAINING_CURVES_PATH
        effective_lr = args.lr if args.lr is not None else DEFAULT_LEARNING_RATE

    model_output = args.model_output if args.model_output is not None else default_model_output
    history_json = args.history_json if args.history_json is not None else default_history_json
    history_csv = args.history_csv if args.history_csv is not None else default_history_csv
    plot_output = args.plot_output if args.plot_output is not None else default_plot_output

    print("\n" + "=" * 60)
    print(f"BRAIN TUMOR DETECTION: MODEL TRAINING ({args.model_type.upper()})")
    print("=" * 60)
    print(f"Model Type              : {args.model_type}")
    print(f"Raw Data Directory      : {args.data_dir.resolve()}")
    print(f"Model Output Path       : {model_output.resolve()}")
    print(f"Max Epochs              : {args.epochs}")
    print(f"Batch Size              : {args.batch_size}")
    print(f"Learning Rate           : {effective_lr}")
    print(f"L2 Regularization       : {args.l2_reg}")
    print(f"Dropout Rate            : {args.dropout}")
    print(f"Early Stopping Patience : {args.patience}")
    print(f"Random Seed             : {args.seed}")
    print("-" * 60)

    try:
        _, result = train_baseline_model(
            data_source=args.data_dir,
            model_type=args.model_type,
            model_output_path=model_output,
            history_json_path=history_json,
            history_csv_path=history_csv,
            plot_output_path=plot_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=effective_lr,
            l2_reg=args.l2_reg,
            dropout_rate=args.dropout,
            early_stopping_patience=args.patience,
            reduce_lr_patience=DEFAULT_REDUCE_LR_PATIENCE,
            reduce_lr_factor=DEFAULT_REDUCE_LR_FACTOR,
            seed=args.seed,
            verbose=1,
        )

        print("\n[SUCCESS] Model training and artifact generation completed successfully.\n")
        return 0

    except FileNotFoundError as fnf_err:
        print(f"\n[DATASET ERROR] {fnf_err}", file=sys.stderr)
        print(
            "\nTo provide the dataset, place brain MRI images into 'data/raw/no/' and 'data/raw/yes/', "
            "or specify --data-dir <path>.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"\n[TRAINING FAILED] Unexpected error during training: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
