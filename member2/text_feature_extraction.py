"""
member2/text_feature_extraction.py
Extracts numerical feature representations from preprocessed text using:
1. TF-IDF Vectorizer (word and n-gram representations for fast, robust baselines).
2. Pretrained Multilingual Transformer embedding extractor (IndicBERT / XLM-R).
"""

import sys
import pickle
from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEXT_CONFIG, MODELS_DIR
from member2.text_preprocessing import preprocess_corpus


class TextFeatureExtractor:
    """Extracts features from text using TF-IDF."""

    def __init__(self, max_features: int = None, ngram_range: Tuple[int, int] = None):
        max_feat = max_features or TEXT_CONFIG["max_features"]
        ng_range = ngram_range or TEXT_CONFIG["ngram_range"]

        self.vectorizer = TfidfVectorizer(
            max_features=max_feat,
            ngram_range=ng_range,
            sublinear_tf=TEXT_CONFIG.get("sublinear_tf", True),
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False

    def fit(self, raw_texts: List[str]) -> "TextFeatureExtractor":
        """Clean and fit vectorizer on training corpus."""
        cleaned = preprocess_corpus(raw_texts)
        self.vectorizer.fit(cleaned)
        self.is_fitted = True
        return self

    def transform(self, raw_texts: List[str]) -> np.ndarray:
        """Clean and transform text corpus into numerical matrix."""
        if not self.is_fitted:
            raise ValueError("TextFeatureExtractor must be fitted before transforming.")
        cleaned = preprocess_corpus(raw_texts)
        return self.vectorizer.transform(cleaned).toarray()

    def fit_transform(self, raw_texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(raw_texts)
        return self.transform(raw_texts)

    def save(self, filepath: Union[str, Path] = None) -> Path:
        """Save fitted vectorizer to disk."""
        path = Path(filepath) if filepath else MODELS_DIR / "text_vectorizer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Text vectorizer saved to {path}")
        return path

    def load(self, filepath: Union[str, Path] = None) -> "TextFeatureExtractor":
        """Load fitted vectorizer from disk."""
        path = Path(filepath) if filepath else MODELS_DIR / "text_vectorizer.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {path}")
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        return self


if __name__ == "__main__":
    sample_corpus = [
        "Passing semester exams is pure joy!",
        "College library opens at nine am.",
        "Server crashed and lost all my assignments."
    ]
    extractor = TextFeatureExtractor()
    features = extractor.fit_transform(sample_corpus)
    print("Fitted Feature Shape:", features.shape)
    print("Sample Feature Vector (first 5 elements):", features[0, :5])
