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
    """
    Multimodal sentiment classifier utilizing:
    1. Early feature concatenation with StandardScaler normalization (default).
    2. Task 4.1: Late fusion probability blending:
       P_final = alpha * P_text + (1 - alpha) * P_image
       with grid search tuning of alpha on the validation split.
    """

    def __init__(self, c_param: float = None, max_iter: int = None, fusion_method: str = "concat"):
        self.fusion_method = (fusion_method or FUSION_CONFIG.get("fusion_method", "concat")).lower()
        self.c_param = c_param or FUSION_CONFIG["c_param"]
        self.max_iter = max_iter or FUSION_CONFIG["max_iter"]
        self.fusion = MultimodalFeatureFusion()
        self.model = LogisticRegression(
            C=self.c_param,
            max_iter=self.max_iter,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
        self.alpha = 0.5
        self.text_model = None
        self.image_model = None
        self.is_trained = False

    def _get_or_load_unimodal_models(self):
        """Helper to ensure text and image models are available for late fusion."""
        from member2.text_model import TextSentimentClassifier
        from member3.image_model import ImageSentimentClassifier
        if self.text_model is None:
            self.text_model = TextSentimentClassifier().load()
        if self.image_model is None:
            self.image_model = ImageSentimentClassifier().load()

    def train(
        self,
        texts: List[str],
        image_paths: List[Union[str, Path]],
        labels: List[int]
    ) -> "MultimodalSentimentClassifier":
        """Fit fusion extractors and train multimodal classifier."""
        if self.fusion_method == "late":
            print(f"[Task 4.1] Training Decision-Level Late Fusion (alpha blending) on {len(texts)} samples...")
            self._get_or_load_unimodal_models()
            
            # Grid search for optimal alpha on validation split
            val_csv = PROCESSED_DATA_DIR / "val.csv"
            best_alpha = 0.5
            best_f1 = -1.0
            
            if val_csv.exists():
                df_val = pd.read_csv(val_csv)
                val_texts = df_val["text"].tolist()
                val_imgs = df_val["image_path"].tolist()
                y_val = df_val["label"].to_numpy()

                t_preds = self.text_model.predict(val_texts)
                i_preds = self.image_model.predict(val_imgs)

                P_t = np.array([[p["probabilities"][c] for c in CLASS_NAMES] for p in t_preds])
                P_i = np.array([[p["probabilities"][c] for c in CLASS_NAMES] for p in i_preds])

                print("Optimizing modality weight alpha in [0.1, 0.9] on validation split:")
                for candidate_alpha in np.linspace(0.1, 0.9, 9):
                    blended = candidate_alpha * P_t + (1.0 - candidate_alpha) * P_i
                    preds = np.argmax(blended, axis=1)
                    score = float(f1_score(y_val, preds, average="macro", zero_division=0))
                    print(f"  alpha={candidate_alpha:.2f} -> Validation Macro-F1: {score:.4f}")
                    if score > best_f1:
                        best_f1 = score
                        best_alpha = round(float(candidate_alpha), 2)
            
            self.alpha = best_alpha
            print(f"Optimal late fusion alpha selected: {self.alpha} (Val Macro-F1: {best_f1:.4f})")
            self.is_trained = True
            return self

        # Default: Early feature concatenation
        print(f"Preparing multimodal features (early concat) for {len(texts)} samples...")
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

        # Late fusion probability blending branch
        if self.fusion_method == "late":
            self._get_or_load_unimodal_models()
            t_preds = self.text_model.predict(texts)
            i_preds = self.image_model.predict(images)

            results = []
            for t_p, i_p in zip(t_preds, i_preds):
                prob_dict = {}
                for c in CLASS_NAMES:
                    p_t = t_p["probabilities"].get(c, 0.0)
                    p_i = i_p["probabilities"].get(c, 0.0)
                    prob_dict[c] = round(float(self.alpha * p_t + (1.0 - self.alpha) * p_i), 4)

                best_cls = max(prob_dict, key=prob_dict.get)
                label_int = CLASS_NAMES.index(best_cls)
                confidence = float(prob_dict[best_cls])

                results.append({
                    "label": label_int,
                    "sentiment": best_cls,
                    "confidence": round(confidence, 4),
                    "probabilities": prob_dict,
                    "fusion_method": "late_fusion",
                    "alpha": self.alpha
                })
            return results

        # Early fusion branch
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
                "probabilities": prob_dict,
                "fusion_method": "concat"
            })
        return results

    def save(self, model_path: Union[str, Path] = None) -> None:
        """Save classifier and fusion pipeline."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "multimodal_model.pkl"
        m_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "fusion_method": self.fusion_method,
            "alpha": self.alpha,
            "model": self.model
        }
        with open(m_path, "wb") as f:
            pickle.dump(save_dict, f)
        
        # Save late fusion config as JSON as well
        config_path = MODELS_DIR / "fusion_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"fusion_method": self.fusion_method, "alpha": self.alpha}, f, indent=2)

        if self.fusion_method != "late":
            self.fusion.save()
        print(f"Multimodal model saved to {m_path} (Method: {self.fusion_method})")

    def load(self, model_path: Union[str, Path] = None) -> "MultimodalSentimentClassifier":
        """Load classifier and fusion pipeline."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "multimodal_model.pkl"
        if not m_path.exists():
            raise FileNotFoundError(f"Multimodal model file not found at {m_path}")

        with open(m_path, "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict) and "fusion_method" in loaded:
            self.fusion_method = loaded["fusion_method"]
            self.alpha = loaded.get("alpha", 0.5)
            self.model = loaded["model"]
        else:
            # Backward compatibility with legacy pickled LogisticRegression
            self.fusion_method = "concat"
            self.model = loaded

        if self.fusion_method != "late":
            self.fusion.load()
        else:
            self._get_or_load_unimodal_models()

        self.is_trained = True
        return self


def train_multimodal_pipeline(fusion_method: str = "concat") -> MultimodalSentimentClassifier:
    """Load train.csv and train multimodal sentiment model."""
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found at {train_csv}. Run Member 1 first.")

    df_train = pd.read_csv(train_csv)
    classifier = MultimodalSentimentClassifier(fusion_method=fusion_method)
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
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Model Training & Evaluation (Member 4)")
    parser.add_argument("--fusion-method", choices=["concat", "late"], default="concat", help="Early concat or decision-level late fusion")
    args = parser.parse_args()

    train_multimodal_pipeline(fusion_method=args.fusion_method)
    evaluate_multimodal_model()
