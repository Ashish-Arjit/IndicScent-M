"""
member3/image_model.py
Trains, saves, and executes inference for the image-only sentiment classifier.
Uses transfer learning on top of ResNet feature representations.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMAGE_CONFIG,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REVERSE_LABEL_MAP,
    CLASS_NAMES,
    RANDOM_SEED
)
from member3.image_feature_extraction import ImageFeatureExtractor


class ImageSentimentClassifier:
    """Image-only sentiment classifier using ResNet features and a linear classification head."""

    def __init__(self, c_param: float = None, max_iter: int = None):
        self.c_param = c_param or IMAGE_CONFIG["c_param"]
        self.max_iter = max_iter or IMAGE_CONFIG["max_iter"]
        self.feature_extractor = ImageFeatureExtractor()
        self.model = LogisticRegression(
            C=self.c_param,
            max_iter=self.max_iter,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
        self.is_trained = False

    def train(self, image_paths: List[Union[str, Path]], labels: List[int]) -> "ImageSentimentClassifier":
        """Extract ResNet features and fit classifier."""
        print(f"Extracting ResNet features for {len(image_paths)} training images...")
        X = self.feature_extractor.extract_features_batch(image_paths)
        y = np.array(labels)
        print("Training Image Classifier head...")
        self.model.fit(X, y)
        self.is_trained = True
        print("Image Classifier training complete.")
        return self

    def predict(
        self,
        image_inputs: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]
    ) -> List[Dict[str, Any]]:
        """Predict sentiment label, class name, and confidence probabilities for images."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Load or train before predicting.")

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        features = []
        for img_in in image_inputs:
            f = self.feature_extractor.extract_features_single(img_in)
            if f is None:
                f = np.zeros(self.feature_extractor.feature_dim, dtype=np.float32)
            features.append(f)

        X = np.array(features, dtype=np.float32)
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
        """Save trained image classifier head."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "image_model.pkl"
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Image model saved to {m_path}")

    def load(self, model_path: Union[str, Path] = None) -> "ImageSentimentClassifier":
        """Load trained image classifier head."""
        m_path = Path(model_path) if model_path else MODELS_DIR / "image_model.pkl"
        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found at {m_path}")

        with open(m_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        return self


def train_image_pipeline() -> ImageSentimentClassifier:
    """Load train.csv and train image sentiment model."""
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found at {train_csv}. Run Member 1 first.")

    df_train = pd.read_csv(train_csv)
    classifier = ImageSentimentClassifier()
    classifier.train(df_train["image_path"].tolist(), df_train["label"].tolist())
    classifier.save()
    return classifier


if __name__ == "__main__":
    train_image_pipeline()
