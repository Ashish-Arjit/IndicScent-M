"""
member4/conflict_detection.py
Detects, diagnoses, and analyzes disagreements between text-only and image-only predictions.
Identifies potential meme sarcasm, mixed signals, and explains confidence disparities.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, FUSION_CONFIG


def detect_modality_conflict(
    text_result: Dict[str, Any],
    image_result: Dict[str, Any],
    multimodal_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare text-only and image-only predictions.
    Diagnose conflict status, confidence differential, and resolution behavior.
    """
    text_sent = text_result.get("sentiment", "unknown")
    text_conf = float(text_result.get("confidence", 0.0))
    image_sent = image_result.get("sentiment", "unknown")
    image_conf = float(image_result.get("confidence", 0.0))

    # Agreement check
    is_conflict = (text_sent != image_sent)
    confidence_diff = round(abs(text_conf - image_conf), 4)
    threshold = FUSION_CONFIG.get("conflict_confidence_threshold", 0.45)

    severity = "NONE"
    explanation = "Modality consensus: Both text and image predict identical sentiment."

    if is_conflict:
        if text_conf >= threshold and image_conf >= threshold:
            severity = "HIGH"
            explanation = (
                f"Strong Modality Clash: Text indicates '{text_sent}' ({text_conf:.2f}), "
                f"whereas Image indicates '{image_sent}' ({image_conf:.2f}). "
                "Often indicates sarcastic memes, ironic humor, or mismatched captions."
            )
        else:
            severity = "MODERATE"
            explanation = (
                f"Moderate Disagreement: Text indicates '{text_sent}' ({text_conf:.2f}), "
                f"Image indicates '{image_sent}' ({image_conf:.2f}). One or both modalities have lower certainty."
            )

    resolution = None
    if multimodal_result is not None:
        mm_sent = multimodal_result.get("sentiment", "unknown")
        mm_conf = float(multimodal_result.get("confidence", 0.0))

        if not is_conflict:
            resolution = f"Consensus maintained by multimodal model ({mm_sent})."
        else:
            if mm_sent == text_sent:
                resolution = f"Multimodal model prioritized text modality ({mm_sent}, {mm_conf:.2f})."
            elif mm_sent == image_sent:
                resolution = f"Multimodal model prioritized visual modality ({mm_sent}, {mm_conf:.2f})."
            else:
                resolution = f"Multimodal model synthesized a third perspective: '{mm_sent}' ({mm_conf:.2f})."

    return {
        "is_conflict": is_conflict,
        "conflict_severity": severity,
        "text_sentiment": text_sent,
        "text_confidence": text_conf,
        "image_sentiment": image_sent,
        "image_confidence": image_conf,
        "confidence_difference": confidence_diff,
        "explanation": explanation,
        "multimodal_resolution": resolution
    }


def analyze_test_set_conflicts() -> Dict[str, Any]:
    """
    Run conflict analysis across the entire test set by joining predictions
    from text_predictions.csv, image_predictions.csv, and multimodal_predictions.csv.
    """
    text_p = RESULTS_DIR / "text_predictions.csv"
    image_p = RESULTS_DIR / "image_predictions.csv"
    mm_p = RESULTS_DIR / "multimodal_predictions.csv"

    if not (text_p.exists() and image_p.exists()):
        print("Predictions CSVs not found in results/. Run evaluation scripts first.")
        return {}

    df_text = pd.read_csv(text_p)
    df_image = pd.read_csv(image_p)
    df_mm = pd.read_csv(mm_p) if mm_p.exists() else None

    # Merge on sample_id
    merged = df_text[["sample_id", "text", "sentiment", "predicted_sentiment", "text_confidence"]].rename(
        columns={"predicted_sentiment": "text_pred", "sentiment": "true_sentiment"}
    )
    merged = merged.merge(
        df_image[["sample_id", "predicted_sentiment", "image_confidence"]].rename(
            columns={"predicted_sentiment": "image_pred"}
        ),
        on="sample_id"
    )

    if df_mm is not None:
        merged = merged.merge(
            df_mm[["sample_id", "predicted_sentiment", "multimodal_confidence"]].rename(
                columns={"predicted_sentiment": "mm_pred"}
            ),
            on="sample_id"
        )

    conflict_records = []
    for _, row in merged.iterrows():
        t_res = {"sentiment": row["text_pred"], "confidence": row["text_confidence"]}
        i_res = {"sentiment": row["image_pred"], "confidence": row["image_confidence"]}
        m_res = {"sentiment": row.get("mm_pred"), "confidence": row.get("multimodal_confidence", 0.0)} if df_mm is not None else None

        diag = detect_modality_conflict(t_res, i_res, m_res)
        row_dict = row.to_dict()
        row_dict.update({
            "is_conflict": diag["is_conflict"],
            "conflict_severity": diag["conflict_severity"],
            "confidence_difference": diag["confidence_difference"],
            "explanation": diag["explanation"],
            "resolution": diag["multimodal_resolution"]
        })
        conflict_records.append(row_dict)

    df_conflicts = pd.DataFrame(conflict_records)
    out_csv = RESULTS_DIR / "conflict_analysis.csv"
    df_conflicts.to_csv(out_csv, index=False)

    total_samples = len(df_conflicts)
    conflicts_count = int(df_conflicts["is_conflict"].sum())
    conflict_rate = round(float(conflicts_count / total_samples * 100), 2) if total_samples > 0 else 0.0

    summary = {
        "total_test_samples": total_samples,
        "disagreeing_samples": conflicts_count,
        "conflict_rate_percent": conflict_rate,
        "high_severity_conflicts": int((df_conflicts["conflict_severity"] == "HIGH").sum()),
        "moderate_conflicts": int((df_conflicts["conflict_severity"] == "MODERATE").sum())
    }

    out_json = RESULTS_DIR / "conflict_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ CONFLICT DETECTION SUMMARY ================")
    print(f"Total Test Samples Checked:    {total_samples}")
    print(f"Modality Disagreements:        {conflicts_count} ({conflict_rate}%)")
    print(f"High-Severity Conflicts:       {summary['high_severity_conflicts']}")
    print(f"Moderate Conflicts:            {summary['moderate_conflicts']}")
    print(f"Detailed analysis saved to:    {out_csv}")
    print("============================================================\n")

    return summary


if __name__ == "__main__":
    analyze_test_set_conflicts()
