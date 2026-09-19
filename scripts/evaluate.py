"""CLI entrypoint for held-out test evaluation and medical performance benchmarking."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASELINE_MODEL_PATH,
    DECISION_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TEST_MANIFEST_PATH,
    DEFAULT_TEST_METRICS_CSV_PATH,
    DEFAULT_TEST_METRICS_JSON_PATH,
    DEFAULT_VAL_MANIFEST_PATH,
    REPORTS_FIGURES_DIR,
)
from src.evaluation.evaluator import (
    evaluate_model_on_manifest,
    evaluate_with_validation_threshold_tuning,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained brain tumor detection model on a held-out test dataset partition."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BASELINE_MODEL_PATH,
        help=f"Path to the trained .keras model file. Default: {BASELINE_MODEL_PATH}",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DEFAULT_TEST_MANIFEST_PATH,
        help=f"Path to the test partition manifest CSV. Default: {DEFAULT_TEST_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_THRESHOLD,
        help=f"Fixed decision threshold in [0.0, 1.0]. Default: {DECISION_THRESHOLD}",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune the decision threshold on the validation split prior to test evaluation.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=DEFAULT_VAL_MANIFEST_PATH,
        help=f"Path to validation manifest CSV (required if --tune-threshold is set). Default: {DEFAULT_VAL_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--tuning-metric",
        type=str,
        default="youden_j",
        choices=["youden_j", "f1", "sensitivity_at_high_specificity"],
        help="Validation metric to optimize if --tune-threshold is enabled. Default: youden_j",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_TEST_METRICS_JSON_PATH,
        help=f"Path to output metrics JSON file. Default: {DEFAULT_TEST_METRICS_JSON_PATH}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_TEST_METRICS_CSV_PATH,
        help=f"Path to output metrics CSV file. Default: {DEFAULT_TEST_METRICS_CSV_PATH}",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPORTS_FIGURES_DIR,
        help=f"Directory to save evaluation plots. Default: {REPORTS_FIGURES_DIR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation. Default: {DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args()


def main() -> int:
    """Run model evaluation pipeline."""
    args = parse_args()

    print("\n" + "=" * 64)
    print("BRAIN TUMOR DETECTION: HELD-OUT TEST BENCHMARK EVALUATION")
    print("=" * 64)
    print(f"Model Artifact Location : {args.model_path.resolve()}")
    print(f"Test Manifest Location  : {args.test_manifest.resolve()}")
    print(f"Decision Threshold Mode : {'Validation-Tuned (' + args.tuning_metric + ')' if args.tune_threshold else f'Fixed ({args.threshold})'}")
    print(f"Output Metrics JSON     : {args.output_json.resolve()}")
    print(f"Output Metrics CSV      : {args.output_csv.resolve()}")
    print(f"Output Figures Directory: {args.figures_dir.resolve()}")
    print("-" * 64 + "\n")

    try:
        if args.tune_threshold:
            report, figure_paths, _, chosen_thresh = evaluate_with_validation_threshold_tuning(
                model=args.model_path,
                val_manifest=args.val_manifest,
                test_manifest=args.test_manifest,
                metric_for_threshold=args.tuning_metric,
                model_name=args.model_path.stem,
                batch_size=args.batch_size,
                metrics_json_path=args.output_json,
                metrics_csv_path=args.output_csv,
                figures_dir=args.figures_dir,
                verbose=1,
            )
        else:
            report, figure_paths, _ = evaluate_model_on_manifest(
                model=args.model_path,
                manifest=args.test_manifest,
                threshold=args.threshold,
                threshold_strategy="fixed_default_0.5" if args.threshold == 0.5 else f"fixed_custom_{args.threshold}",
                model_name=args.model_path.stem,
                split_name="test",
                batch_size=args.batch_size,
                metrics_json_path=args.output_json,
                metrics_csv_path=args.output_csv,
                figures_dir=args.figures_dir,
                generate_plots=True,
                verbose=1,
            )

        print("\n[SUCCESS] Test set evaluation and diagnostic reporting completed successfully.\n")
        return 0

    except FileNotFoundError as fnf_err:
        print(f"\n[EVALUATION ERROR] {fnf_err}", file=sys.stderr)
        print(
            "\nPlease ensure that the model is trained (`python scripts/train.py`) and that "
            "test manifests exist (`python scripts/prepare_data.py`).",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"\n[EVALUATION FAILED] Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
