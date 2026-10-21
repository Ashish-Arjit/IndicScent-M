"""
member4/feature_fusion.py
Combines text and image feature embeddings into unified multimodal representations.
Provides feature normalization and concatenation across both modalities.
"""

import sys
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Optional
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, FUSION_CONFIG
from member2.text_feature_extraction import TextFeatureExtractor
from member3.image_feature_extraction import ImageFeatureExtractor


class MultimodalFeatureFusion:
    """Concatenates and scales text features and image features into a unified embedding."""

    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, texts: List[str], image_paths: List[Union[str, Path]]) -> "MultimodalFeatureFusion":
        """Fit text vectorizer and feature scaler on paired training data."""
        print("Extracting text features for fusion...")
        text_feats = self.text_extractor.fit_transform(texts)

        print("Extracting image features for fusion...")
        image_feats = self.image_extractor.extract_features_batch(image_paths)

        # Early fusion: Concatenate representations along feature axis
        combined = np.hstack([text_feats, image_feats])
        self.scaler.fit(combined)
        self.is_fitted = True
        return self

    def transform(
        self,
        texts: List[str],
        image_paths: List[Union[str, Path, Image.Image]]
    ) -> np.ndarray:
        """Transform text and image inputs into scaled multimodal feature vectors."""
        if not self.is_fitted:
            raise ValueError("Fusion module must be fitted before transform.")

        text_feats = self.text_extractor.transform(texts)
        
        image_feats = []
        for img in image_paths:
            f = self.image_extractor.extract_features_single(img)
            if f is None:
                f = np.zeros(self.image_extractor.feature_dim, dtype=np.float32)
            image_feats.append(f)
        image_feats = np.array(image_feats, dtype=np.float32)

        combined = np.hstack([text_feats, image_feats])
        return self.scaler.transform(combined)

    def extract_single(
        self,
        text: str,
        image_input: Union[str, Path, Image.Image]
    ) -> np.ndarray:
        """Extract a single multimodal feature vector."""
        return self.transform([text], [image_input])

    def save(self, filepath: Union[str, Path] = None) -> None:
        """Save fitted fusion scaler and associated sub-vectorizers."""
        s_path = Path(filepath) if filepath else MODELS_DIR / "fusion_scaler.pkl"
        s_path.parent.mkdir(parents=True, exist_ok=True)
        with open(s_path, "wb") as f:
            pickle.dump(self.scaler, f)
        self.text_extractor.save()
        print(f"Fusion scaler saved to {s_path}")

    def load(self, filepath: Union[str, Path] = None) -> "MultimodalFeatureFusion":
        """Load fitted fusion scaler and sub-vectorizers."""
        s_path = Path(filepath) if filepath else MODELS_DIR / "fusion_scaler.pkl"
        if not s_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {s_path}")
        with open(s_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.text_extractor.load()
        self.is_fitted = True
        return self


if __name__ == "__main__":
    sample_texts = ["Happy day!", "Sad accident."]
    dummy_imgs = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]
    fusion = MultimodalFeatureFusion()
    fusion.fit(sample_texts, dummy_imgs)
    vec = fusion.extract_single("Happy day!", dummy_imgs[0])
    print("Fused Multimodal Feature Vector Shape:", vec.shape)
