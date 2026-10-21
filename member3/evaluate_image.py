"""
member3/evaluate_image.py
Evaluates the trained image sentiment model on the test split.
Calculates Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix.
Saves metrics and individual test sample predictions to results/.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR, CLASS_NAMES, REVERSE_LABEL_MAP
from member3.image_model import ImageSentimentClassifier


def evaluate_image_model() -> Dict[str, Any]:
    """Evaluate image model on test.csv."""
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found at {test_csv}. Run Member 1 first.")

    df_test = pd.read_csv(test_csv)
    classifier = ImageSentimentClassifier()
    classifier.load()

    # Predict on test set
    predictions = classifier.predict(df_test["image_path"].tolist())

    y_true = df_test["label"].to_numpy()
    y_pred = np.array([p["label"] for p in predictions])
    confidences = [p["confidence"] for p in predictions]

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    metrics = {
        "model_name": "Image-Only (ResNet-18 + Logistic Regression)",
        "modality": "image",
        "num_test_samples": len(df_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision_macro": round(cls_report["macro avg"]["precision"], 4),
        "recall_macro": round(cls_report["macro avg"]["recall"], 4),
        "confusion_matrix": cm,
        "class_report": cls_report
    }

    # Save metrics JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed predictions CSV
    df_preds = df_test.copy()
    df_preds["predicted_label"] = y_pred
    df_preds["predicted_sentiment"] = [REVERSE_LABEL_MAP[l] for l in y_pred]
    df_preds["image_confidence"] = confidences
    df_preds["is_correct"] = (y_true == y_pred)
    df_preds.to_csv(RESULTS_DIR / "image_predictions.csv", index=False)

    print("\n============== IMAGE-ONLY MODEL EVALUATION ==============")
    print(f"Accuracy:       {acc * 100:.2f}%")
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print(f"Precision:      {cls_report['macro avg']['precision']:.4f}")
    print(f"Recall:         {cls_report['macro avg']['recall']:.4f}")
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    print("      Neg  Neu  Pos")
    for idx, row in enumerate(cm):
        print(f" {CLASS_NAMES[idx][:3]:>3} {row[0]:>4} {row[1]:>4} {row[2]:>4}")
    print("=========================================================\n")

    return metrics


if __name__ == "__main__":
    evaluate_image_model()
