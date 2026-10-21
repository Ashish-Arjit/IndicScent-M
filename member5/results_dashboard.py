"""
member5/results_dashboard.py
Console and visual dashboard for inspecting all project benchmarks,
confusion matrices, and conflict statistics.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR


def print_project_dashboard():
    """Print complete summary of dataset, models, and conflict detection."""
    print("\n" + "=" * 70)
    print("              INDICSENT-M PROJECT SUMMARY DASHBOARD           ")
    print("=" * 70)

    # 1. Dataset Quality
    dq_file = RESULTS_DIR / "data_quality_report.json"
    if dq_file.exists():
        with open(dq_file, "r", encoding="utf-8") as f:
            dq = json.load(f)
        print("\n[DATASET INTEGRITY]")
        print(f"Status:             {dq.get('integrity_status', 'UNKNOWN')}")
        print(f"Total Clean Pairs:  {dq['dataset_inspection']['total_rows']}")
        print(f"Missing Images:     {dq['image_text_pair_validation']['missing_images_count']}")
        print(f"Average Words/Post: {dq['text_statistics']['mean_words']}")
    else:
        print("\n[DATASET INTEGRITY] Not evaluated yet.")

    # 2. Model Performance
    comp_file = RESULTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        print("\n[MODEL BENCHMARK COMPARISON]")
        print(df_comp.to_string(index=False))
    else:
        print("\n[MODEL BENCHMARK COMPARISON] Not evaluated yet.")

    # 3. Conflict Detection
    conf_file = RESULTS_DIR / "conflict_summary.json"
    if conf_file.exists():
        with open(conf_file, "r", encoding="utf-8") as f:
            conf = json.load(f)
        print("\n[CONFLICT & SARCASM DETECTION]")
        print(f"Test Samples Analyzed:   {conf.get('total_test_samples', 0)}")
        print(f"Modality Disagreements:  {conf.get('disagreeing_samples', 0)} ({conf.get('conflict_rate_percent', 0)}%)")
        print(f"High-Severity Conflicts: {conf.get('high_severity_conflicts', 0)}")
    else:
        print("\n[CONFLICT & SARCASM DETECTION] Not evaluated yet.")

    print("\n" + "=" * 70)
    print("To launch interactive UI: streamlit run member5/app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_project_dashboard()
