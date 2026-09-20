"""Multi-model benchmarking and comparison utilities."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.config import (
    DEFAULT_MODEL_COMPARISON_CSV_PATH,
    DEFAULT_MODEL_COMPARISON_JSON_PATH,
)


@dataclass
class ModelComparisonEntry:
    """Benchmark metrics record for a single model configuration."""

    model_name: str
    architecture_type: str
    split: str
    test_set_usage: str
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    best_validation_loss: float
    best_validation_epoch: int
    test_threshold: float
    test_threshold_strategy: str
    accuracy: float
    precision: float
    sensitivity: float  # Recall / TPR
    specificity: float  # TNR
    f1_score: float
    roc_auc: Optional[float]
    average_precision: Optional[float]  # PR-AUC
    tp: int
    tn: int
    fp: int
    fn: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return asdict(self)


def generate_comparison_report(
    entries: List[ModelComparisonEntry],
    output_json_path: Union[str, Path] = DEFAULT_MODEL_COMPARISON_JSON_PATH,
    output_csv_path: Union[str, Path] = DEFAULT_MODEL_COMPARISON_CSV_PATH,
) -> Tuple[Path, Path, pd.DataFrame]:
    """Generate and save multi-model benchmark comparison in JSON and CSV formats.

    Args:
        entries: List of ModelComparisonEntry instances.
        output_json_path: Destination path for JSON report.
        output_csv_path: Destination path for CSV report.

    Returns:
        Tuple of (saved_json_path, saved_csv_path, comparison_dataframe).
    """
    json_path = Path(output_json_path)
    csv_path = Path(output_csv_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    records = [entry.to_dict() for entry in entries]
    df = pd.DataFrame(records)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Save CSV
    df.to_csv(csv_path, index=False)

    return json_path, csv_path, df
