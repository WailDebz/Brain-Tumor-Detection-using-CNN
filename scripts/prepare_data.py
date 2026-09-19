"""CLI script to scan, validate, deduplicate, and create leak-free dataset splits."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SPLIT_RATIOS, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_DATA_DIR
from src.data.ingestion import scan_and_ingest_data
from src.data.split import create_stratified_splits, save_splits


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan, validate, and generate leak-free stratified dataset splits for Brain Tumor Detection."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory containing raw class subfolders ('no', 'yes'). Default: {RAW_DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Output directory to save split manifest CSVs. Default: {PROCESSED_DATA_DIR}",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[0],
        help=f"Proportion of data for training. Default: {DEFAULT_SPLIT_RATIOS[0]}",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[1],
        help=f"Proportion of data for validation. Default: {DEFAULT_SPLIT_RATIOS[1]}",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[2],
        help=f"Proportion of data for test set. Default: {DEFAULT_SPLIT_RATIOS[2]}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for deterministic splitting. Default: {RANDOM_SEED}",
    )
    return parser.parse_args()


def main() -> int:
    """Run data ingestion, validation, splitting, and manifest generation."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("BRAIN TUMOR DETECTION: DATA INGESTION & LEAK-FREE SPLITTING")
    print("=" * 60)
    print(f"Target Raw Data Directory : {args.data_dir.resolve()}")
    print(f"Output Manifest Directory : {args.output_dir.resolve()}")
    print(f"Split Ratios (Train/Val/Test): {args.train_ratio:.2f} / {args.val_ratio:.2f} / {args.test_ratio:.2f}")
    print(f"Random Seed               : {args.seed}")
    print("-" * 60)

    try:
        # Step 1: Scan and Ingest Data
        print("\n[1/3] Scanning and validating raw image dataset...")
        df, ingestion_report = scan_and_ingest_data(args.data_dir)
        print(ingestion_report.summary())

        if ingestion_report.corrupt_files:
            print("\n[WARNING] Corrupt or unreadable files skipped:")
            for p, err in ingestion_report.corrupt_files:
                print(f"  - {p}: {err}")

        if ingestion_report.duplicate_files:
            print("\n[WARNING] Duplicate files skipped based on SHA-256 hash matching:")
            for p, orig in ingestion_report.duplicate_files:
                print(f"  - {p} (Identical to: {orig})")

        # Step 2: Stratified Split on Original Images
        print("\n[2/3] Performing leak-free stratified splitting on original samples...")
        split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
        train_df, val_df, test_df, split_report = create_stratified_splits(
            df=df,
            split_ratios=split_ratios,
            seed=args.seed,
        )
        print(split_report.summary())

        # Step 3: Save Manifests
        print("\n[3/3] Saving split manifests to disk...")
        saved_paths = save_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=args.output_dir,
        )

        for split_name, path in saved_paths.items():
            print(f"  - {split_name.capitalize():10s} manifest: {path.resolve()}")

        print("\n[SUCCESS] Data preparation and leak-free split generation completed successfully.\n")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] Data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
