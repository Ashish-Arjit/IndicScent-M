"""
member2/text_model.py
Trains, saves, and executes inference for the text-only sentiment classifier.
Provides calibrated probability distributions across Positive, Neutral, and Negative.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TEXT_CONFIG,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REVERSE_LABEL_MAP,
    CLASS_NAMES,
    RANDOM_SEED
)
from member2.text_feature_extraction import TextFeatureExtractor


class TextSentimentClassifier:
    """Text-only sentiment classification pipeline combining vectorizer and classifier."""

    def __init__(self, c_param: float = None, max_iter: int = None):
        self.c_param = c_param or TEXT_CONFIG["c_param"]
        self.max_iter = max_iter or TEXT_CONFIG["max_iter"]
        self.feature_extractor = TextFeatureExtractor()
        self.model = LogisticRegression(
            C=self.c_param,
            max_iter=self.max_iter,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
        self.is_trained = False

    def train(self, texts: List[str], labels: List[int]) -> "TextSentimentClassifier":
        """Fit vectorizer and train classifier."""
        print(f"Training Text Classifier on {len(texts)} samples...")
        X = self.feature_extractor.fit_transform(texts)
        y = np.array(labels)
        self.model.fit(X, y)
        self.is_trained = True
        print("Text Classifier training complete.")
        return self

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Predict sentiment label, class name, and confidence probabilities."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Load or train before predicting.")

        if isinstance(texts, str):
            texts = [texts]

        X = self.feature_extractor.transform(texts)
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

    def save(self, model_path: Union[str, Path] = None, vec_path: Union[str, Path] = None) -> None:
        """Save classifier and vectorizer."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "text_model.pkl"
        v_path = Path(vec_path) if vec_path else MODELS_DIR / "text_vectorizer.pkl"

        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "wb") as f:
            pickle.dump(self.model, f)
        self.feature_extractor.save(v_path)
        print(f"Text model saved to {m_path}")

    def load(self, model_path: Union[str, Path] = None, vec_path: Union[str, Path] = None) -> "TextSentimentClassifier":
        """Load classifier and vectorizer."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "text_model.pkl"
        v_path = Path(vec_path) if vec_path else MODELS_DIR / "text_vectorizer.pkl"

        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found at {m_path}")

        with open(m_path, "rb") as f:
            self.model = pickle.load(f)
        self.feature_extractor.load(v_path)
        self.is_trained = True
        return self


def train_text_pipeline() -> TextSentimentClassifier:
    """Load train.csv and train text sentiment model."""
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found at {train_csv}. Run Member 1 first.")

    df_train = pd.read_csv(train_csv)
    classifier = TextSentimentClassifier()
    classifier.train(df_train["text"].tolist(), df_train["label"].tolist())
    classifier.save()
    return classifier


if __name__ == "__main__":
    train_text_pipeline()
