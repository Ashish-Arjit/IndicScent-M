"""
member4/multimodal_model.py
Trains, saves, and evaluates the multimodal sentiment classifier.
Fuses text and visual representations and evaluates test performance.
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FUSION_CONFIG,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    REVERSE_LABEL_MAP,
    CLASS_NAMES,
    RANDOM_SEED
)
from member4.feature_fusion import MultimodalFeatureFusion


class MultimodalSentimentClassifier:
    """Multimodal sentiment classifier utilizing concatenated text and visual representations."""

    def __init__(self, c_param: float = None, max_iter: int = None):
        self.c_param = c_param or FUSION_CONFIG["c_param"]
        self.max_iter = max_iter or FUSION_CONFIG["max_iter"]
        self.fusion = MultimodalFeatureFusion()
        self.model = LogisticRegression(
            C=self.c_param,
            max_iter=self.max_iter,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
        self.is_trained = False

    def train(
        self,
        texts: List[str],
        image_paths: List[Union[str, Path]],
        labels: List[int]
    ) -> "MultimodalSentimentClassifier":
        """Fit fusion extractors and train multimodal classifier."""
        print(f"Preparing multimodal features for {len(texts)} samples...")
        self.fusion.fit(texts, image_paths)
        X = self.fusion.transform(texts, image_paths)
        y = np.array(labels)

        print("Training Multimodal Classifier...")
        self.model.fit(X, y)
        self.is_trained = True
        print("Multimodal Classifier training complete.")
        return self

    def predict(
        self,
        texts: Union[str, List[str]],
        images: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]
    ) -> List[Dict[str, Any]]:
        """Predict sentiment using combined text and image inputs."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Load or train before predicting.")

        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(images, list):
            images = [images]

        if len(texts) != len(images):
            raise ValueError("Texts and images lists must have identical lengths.")

        X = self.fusion.transform(texts, images)
        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)

        results = []
        for pred, prob in zip(preds, probas):
            label_int = int(pred)
            sentiment_name = REVERSE_LABEL_MAP.get(label_int, "unknown")
            confidence = float(np.max(prob))
            prob_dict = {CLASS_NAMES[i]: round(float(prob[i]), 4) for i in range(len(CLASS_NAMES))}

            results.append({
                "label": label_int,
                "sentiment": sentiment_name,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict
            })
        return results

    def save(self, model_path: Union[str, Path] = None) -> None:
        """Save classifier and fusion pipeline."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "multimodal_model.pkl"
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "wb") as f:
            pickle.dump(self.model, f)
        self.fusion.save()
        print(f"Multimodal model saved to {m_path}")

    def load(self, model_path: Union[str, Path] = None) -> "MultimodalSentimentClassifier":
        """Load classifier and fusion pipeline."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "multimodal_model.pkl"
        if not m_path.exists():
            raise FileNotFoundError(f"Multimodal model file not found at {m_path}")

        with open(m_path, "rb") as f:
            self.model = pickle.load(f)
        self.fusion.load()
        self.is_trained = True
        return self


def train_multimodal_pipeline() -> MultimodalSentimentClassifier:
    """Load train.csv and train multimodal sentiment model."""
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found at {train_csv}. Run Member 1 first.")

    df_train = pd.read_csv(train_csv)
    classifier = MultimodalSentimentClassifier()
    classifier.train(
        texts=df_train["text"].tolist(),
        image_paths=df_train["image_path"].tolist(),
        labels=df_train["label"].tolist()
    )
    classifier.save()
    return classifier


def evaluate_multimodal_model() -> Dict[str, Any]:
    """Evaluate multimodal model on test.csv."""
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found at {test_csv}. Run Member 1 first.")

    df_test = pd.read_csv(test_csv)
    classifier = MultimodalSentimentClassifier()
    classifier.load()

    predictions = classifier.predict(
        texts=df_test["text"].tolist(),
        images=df_test["image_path"].tolist()
    )

    y_true = df_test["label"].to_numpy()
    y_pred = np.array([p["label"] for p in predictions])
    confidences = [p["confidence"] for p in predictions]

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
        "model_name": "Multimodal Fusion (TF-IDF + ResNet-18)",
        "modality": "multimodal",
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
    with open(RESULTS_DIR / "multimodal_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed predictions CSV
    df_preds = df_test.copy()
    df_preds["predicted_label"] = y_pred
    df_preds["predicted_sentiment"] = [REVERSE_LABEL_MAP[l] for l in y_pred]
    df_preds["multimodal_confidence"] = confidences
    df_preds["is_correct"] = (y_true == y_pred)
    df_preds.to_csv(RESULTS_DIR / "multimodal_predictions.csv", index=False)

    print("\n=========== MULTIMODAL MODEL EVALUATION ============")
    print(f"Accuracy:       {acc * 100:.2f}%")
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print(f"Precision:      {cls_report['macro avg']['precision']:.4f}")
    print(f"Recall:         {cls_report['macro avg']['recall']:.4f}")
    print("=====================================================\n")

    return metrics


if __name__ == "__main__":
    train_multimodal_pipeline()
    evaluate_multimodal_model()
