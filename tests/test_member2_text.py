"""
tests/test_member2_text.py
Unit tests for Member 2: Text preprocessing, TF-IDF vectorization, and text model.
"""

import sys
from pathlib import Path
import unittest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from member2.text_preprocessing import clean_text
from member2.text_feature_extraction import TextFeatureExtractor
from member2.text_model import TextSentimentClassifier


class TestMember2Text(unittest.TestCase):
    """Tests for Member 2 text module."""

    def test_clean_text_noise_removal(self):
        """Test URL removal, mention stripping, and repeated character reduction."""
        raw = "Check this out https://example.com @user sooooo good!!!!"
        cleaned = clean_text(raw)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("@user", cleaned)
        self.assertIn("soo good", cleaned)

    def test_text_feature_shape(self):
        """Test vectorizer output dimensions."""
        texts = ["Positive feedback", "Negative review", "Neutral update"]
        extractor = TextFeatureExtractor(max_features=10)
        feats = extractor.fit_transform(texts)
        self.assertEqual(feats.shape[0], 3)
        self.assertLessEqual(feats.shape[1], 10)

    def test_text_model_train_and_predict(self):
        """Test quick train and predict flow."""
        texts = [
            "Loved the festival celebration and fireworks",
            "Terrible customer service and rude staff",
            "Bus arrival schedule updated for afternoon"
        ]
        labels = [2, 0, 1]  # positive, negative, neutral
        clf = TextSentimentClassifier()
        clf.train(texts, labels)
        self.assertTrue(clf.is_trained)

        preds = clf.predict("Very terrible experience")
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0]["sentiment"], ["positive", "neutral", "negative"])
        self.assertIn("probabilities", preds[0])


if __name__ == "__main__":
    unittest.main()
