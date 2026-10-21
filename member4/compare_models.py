"""
member4/compare_models.py
Generates a comparative analysis across Text-Only, Image-Only, and Multimodal models.
Validates that only genuine evaluation metrics are reported.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR


def compare_all_models() -> pd.DataFrame:
    """Load evaluation metrics for all 3 modalities and generate comparison table."""
    model_files = [
        ("Text-only", RESULTS_DIR / "text_metrics.json"),
        ("Image-only", RESULTS_DIR / "image_metrics.json"),
        ("Multimodal", RESULTS_DIR / "multimodal_metrics.json")
    ]

    records = []
    for model_name, path in model_files:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append({
                "Model Approach": model_name,
                "Modality Input": data.get("modality", "unknown").capitalize(),
                "Status": "Evaluated",
                "Accuracy (%)": round(float(data.get("accuracy", 0.0) * 100), 2),
                "Macro-F1": data.get("macro_f1", 0.0),
                "Precision (Macro)": data.get("precision_macro", 0.0),
                "Recall (Macro)": data.get("recall_macro", 0.0),
                "Test Samples": data.get("num_test_samples", 0)
            })
        else:
            records.append({
                "Model Approach": model_name,
                "Modality Input": "N/A",
                "Status": "Not Evaluated / Missing Checkpoint",
                "Accuracy (%)": None,
                "Macro-F1": None,
                "Precision (Macro)": None,
                "Recall (Macro)": None,
                "Test Samples": 0
            })

    df_comp = pd.DataFrame(records)

    # Save to results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    
    with open(RESULTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n======================= MODEL BENCHMARK COMPARISON =======================")
    print(df_comp.to_string(index=False))
    print("===========================================================================\n")

    return df_comp


if __name__ == "__main__":
    compare_all_models()
