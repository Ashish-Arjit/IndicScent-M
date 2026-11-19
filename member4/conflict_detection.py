"""
member4/conflict_detection.py
Detects, diagnoses, and analyzes disagreements between text-only and image-only predictions.
Identifies potential meme sarcasm, mixed signals, and explains confidence disparities.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, PROCESSED_DATA_DIR, FUSION_CONFIG


def detect_modality_conflict(
    text_result: Dict[str, Any],
    image_result: Dict[str, Any],
    multimodal_result: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Confidence-Aware Cross-Modal Sentiment Conflict Detection.
    1. Compares text-only and image-only predictions.
    2. Compares their confidence scores.
    3. Identifies cases where the two modalities disagree.
    4. Flags conflicting predictions with severity level.
    5. Displays the multimodal prediction when available.
    6. Returns structured conflict diagnosis (without anthropomorphic reasoning explanations).
    """
    text_sent = text_result.get("sentiment", "unknown")
    text_conf = float(text_result.get("confidence", 0.0))
    image_sent = image_result.get("sentiment", "unknown")
    image_conf = float(image_result.get("confidence", 0.0))

    # 1. Modality agreement check
    is_conflict = (text_sent != image_sent)
    confidence_diff = round(abs(text_conf - image_conf), 4)
    thresh = threshold if threshold is not None else FUSION_CONFIG.get("conflict_confidence_threshold", 0.45)

    severity = "NONE"
    explanation = f"Modality Consensus: Both text and image independently predict '{text_sent}'."

    # 4. Severity categorization
    if is_conflict:
        if text_conf >= thresh and image_conf >= thresh:
            severity = "HIGH"
            explanation = (
                f"High-Severity Conflict: Text indicates '{text_sent}' (conf: {text_conf:.2f}), "
                f"while Image indicates '{image_sent}' (conf: {image_conf:.2f}). "
                f"Modality confidence differential: {confidence_diff:.2f}."
            )
        else:
            severity = "MODERATE"
            explanation = (
                f"Moderate Disagreement: Text indicates '{text_sent}' (conf: {text_conf:.2f}), "
                f"Image indicates '{image_sent}' (conf: {image_conf:.2f}). One or both modalities show lower certainty."
            )

    # 5. Multimodal resolution display
    mm_sent = None
    mm_conf = None
    resolution = None
    if multimodal_result is not None:
        mm_sent = multimodal_result.get("sentiment", "unknown")
        mm_conf = float(multimodal_result.get("confidence", 0.0))

        if not is_conflict:
            resolution = f"Consensus maintained by multimodal model: '{mm_sent}' ({mm_conf:.2f})."
        else:
            if mm_sent == text_sent:
                resolution = f"Multimodal decision aligned with text modality: '{mm_sent}' ({mm_conf:.2f})."
            elif mm_sent == image_sent:
                resolution = f"Multimodal decision aligned with visual modality: '{mm_sent}' ({mm_conf:.2f})."
            else:
                resolution = f"Multimodal decision synthesized alternative class: '{mm_sent}' ({mm_conf:.2f})."

    return {
        "is_conflict": is_conflict,
        "conflict_severity": severity,
        "text_sentiment": text_sent,
        "text_confidence": text_conf,
        "image_sentiment": image_sent,
        "image_confidence": image_conf,
        "confidence_difference": confidence_diff,
        "multimodal_sentiment": mm_sent,
        "multimodal_confidence": mm_conf,
        "multimodal_resolution": resolution,
        "explanation": explanation
    }


def tune_conflict_threshold(
    val_csv: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Task 4.2: Adaptive Conflict Threshold Tuning & Error Analysis.
    Optimizes the confidence differential threshold to maximize F1-score
    specifically on sarcastic/ironic memes or disagreement detection.
    Evaluates precision, recall, and F1 across candidate thresholds [0.20, 0.65].
    """
    target_csv = val_csv if val_csv else PROCESSED_DATA_DIR / "val.csv"
    if not target_csv.exists():
        print(f"Validation dataset not found at {target_csv}. Using default threshold.")
        return {"tuned_threshold": 0.45}

    df_val = pd.read_csv(target_csv)
    from member2.text_model import TextSentimentClassifier
    from member3.image_model import ImageSentimentClassifier
    from member4.multimodal_model import MultimodalSentimentClassifier

    text_clf = TextSentimentClassifier().load()
    image_clf = ImageSentimentClassifier().load()
    mm_clf = MultimodalSentimentClassifier().load()

    t_preds = text_clf.predict(df_val["text"].tolist())
    i_preds = image_clf.predict(df_val["image_path"].tolist())
    m_preds = mm_clf.predict(df_val["text"].tolist(), df_val["image_path"].tolist())

    candidates = np.linspace(0.20, 0.65, 10).round(2).tolist()
    best_thresh = 0.45
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    curve_data = []

    print("\n[Task 4.2] Adaptive Threshold Tuning (Precision-Recall Optimization):")
    print(f"{'Threshold (tau)':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 54)

    for tau in candidates:
        tp = 0
        fp = 0
        fn = 0
        for idx in range(len(df_val)):
            diag = detect_modality_conflict(t_preds[idx], i_preds[idx], m_preds[idx], threshold=tau)
            # Ground truth disagreement is defined when unimodal predictions diverge
            ground_truth_disagree = (t_preds[idx]["sentiment"] != i_preds[idx]["sentiment"])
            pred_high = (diag["conflict_severity"] == "HIGH")

            if pred_high and ground_truth_disagree:
                tp += 1
            elif pred_high and not ground_truth_disagree:
                fp += 1
            elif not pred_high and ground_truth_disagree:
                fn += 1

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        curve_data.append({
            "threshold": tau,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })
        print(f"tau = {tau:<10.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = tau
            best_p = prec
            best_r = rec

    print("-" * 54)
    print(f"Optimal Adaptive Conflict Threshold: tau* = {best_thresh:.2f} (F1: {best_f1:.4f}, Prec: {best_p:.4f}, Rec: {best_r:.4f})\n")

    tuning_summary = {
        "tuned_threshold": best_thresh,
        "tuning_precision": round(best_p, 4),
        "tuning_recall": round(best_r, 4),
        "tuning_f1": round(best_f1, 4),
        "threshold_curve": curve_data
    }

    # Update results/conflict_summary.json if exists
    sum_json = RESULTS_DIR / "conflict_summary.json"
    existing_sum = {}
    if sum_json.exists():
        with open(sum_json, "r", encoding="utf-8") as f:
            try:
                existing_sum = json.load(f)
            except Exception:
                pass
    existing_sum.update(tuning_summary)
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(existing_sum, f, indent=2)

    return tuning_summary


def analyze_test_set_conflicts() -> Dict[str, Any]:
    """
    Run conflict analysis across the entire test set by joining predictions
    from text_predictions.csv, image_predictions.csv, and multimodal_predictions.csv.
    Saves results to results/conflict_analysis.csv and results/conflict_summary.json.
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
    parser = argparse.ArgumentParser(description="Conflict & Disagreement Analysis (Member 4)")
    parser.add_argument("--tune-threshold", action="store_true", help="Run adaptive threshold ROC optimization (Task 4.2)")
    args = parser.parse_args()

    if args.tune_threshold:
        tune_conflict_threshold()
    else:
        analyze_test_set_conflicts()
