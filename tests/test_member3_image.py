"""
tests/test_member3_image.py
Unit tests for Member 3: Image preprocessing, ResNet feature extraction, and image model.
"""

import sys
from pathlib import Path
import unittest
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from member3.image_preprocessing import load_and_preprocess_image
from member3.image_feature_extraction import ImageFeatureExtractor
from member3.image_model import ImageSentimentClassifier


class TestMember3Image(unittest.TestCase):
    """Tests for Member 3 image module."""

    def test_load_and_preprocess_image(self):
        """Test image loading from PIL object."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        tensor = load_and_preprocess_image(img)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_image_feature_extractor(self):
        """Test ResNet feature dimension extraction."""
        img = Image.new("RGB", (224, 224), color=(60, 120, 180))
        extractor = ImageFeatureExtractor()
        feat = extractor.extract_features_single(img)
        self.assertIsNotNone(feat)
        self.assertEqual(len(feat), 512)

    def test_image_model_train_and_predict(self):
        """Test image classifier head training on dummy images."""
        imgs = [
            Image.new("RGB", (224, 224), color=(0, 255, 0)),
            Image.new("RGB", (224, 224), color=(255, 0, 0)),
            Image.new("RGB", (224, 224), color=(128, 128, 128))
        ]
        labels = [2, 0, 1]
        clf = ImageSentimentClassifier()
        # Train with list of PIL images
        extractor = ImageFeatureExtractor()
        feats = [extractor.extract_features_single(img) for img in imgs]
        clf.model.fit(feats, labels)
        clf.is_trained = True

        preds = clf.predict(imgs[0])
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0]["sentiment"], ["positive", "neutral", "negative"])


if __name__ == "__main__":
    unittest.main()
