"""
member1/label_analysis.py
Analyzes class distributions, checks class balance across train/val/test splits,
and generates class distribution summaries.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR, LABEL_MAP


def analyze_split_distributions() -> Dict[str, Any]:
    """Analyze sentiment label distribution across all processed splits."""
    splits = ["train", "val", "test", "all_clean"]
    distribution_report = {}

    for split in splits:
        csv_path = PROCESSED_DATA_DIR / f"{split}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        counts = df["sentiment"].value_counts().to_dict()
        percentages = (df["sentiment"].value_counts(normalize=True) * 100).round(2).to_dict()

        distribution_report[split] = {
            "total_samples": len(df),
            "counts": {cls: counts.get(cls, 0) for cls in LABEL_MAP.keys()},
            "percentages": {cls: percentages.get(cls, 0.0) for cls in LABEL_MAP.keys()}
        }

    # Save to results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "label_distribution.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(distribution_report, f, indent=2)

    return distribution_report


def print_label_analysis(report: Dict[str, Any]) -> None:
    """Print readable distribution table."""
    print("\n================== LABEL DISTRIBUTION REPORT ==================")
    print(f"{'Split':<12} | {'Negative':<15} | {'Neutral':<15} | {'Positive':<15} | {'Total':<6}")
    print("-" * 72)
    for split, data in report.items():
        neg = f"{data['counts']['negative']} ({data['percentages']['negative']}%)"
        neu = f"{data['counts']['neutral']} ({data['percentages']['neutral']}%)"
        pos = f"{data['counts']['positive']} ({data['percentages']['positive']}%)"
        total = data["total_samples"]
        print(f"{split:<12} | {neg:<15} | {neu:<15} | {pos:<15} | {total:<6}")
    print("===============================================================\n")


if __name__ == "__main__":
    report = analyze_split_distributions()
    print_label_analysis(report)
