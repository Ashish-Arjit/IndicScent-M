"""
member5/testing.py
Integration testing suite for IndicSent-M. Verifies that all 5 member modules
interoperate correctly and adhere to input/output specifications.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LABEL_MAP, CLASS_NAMES
from member1.dataset_preprocessing import clean_and_split_dataset, create_sample_benchmark_dataset
from member2.text_preprocessing import clean_text
from member2.text_feature_extraction import TextFeatureExtractor
from member3.image_preprocessing import load_and_preprocess_image
from member3.image_feature_extraction import ImageFeatureExtractor
from member4.conflict_detection import detect_modality_conflict
from member5.integration import IndicSentPipeline


class TestIndicSentIntegration(unittest.TestCase):
    """Integration test suite for the complete IndicSent-M pipeline."""

    def test_01_text_preprocessing(self):
        """Test emoji preservation and character normalization."""
        raw = "Passed the exam!! Soooo happy 😊 check https://test.edu @friends"
        cleaned = clean_text(raw)
        self.assertIn("emoji_positive", cleaned)
        self.assertNotIn("https://test.edu", cleaned)
        self.assertNotIn("@friends", cleaned)
        self.assertIn("exam", cleaned)

    def test_02_text_feature_extraction(self):
        """Test TF-IDF feature extraction dimensions and fitting."""
        corpus = ["Good day today", "Bad server crash", "Library open"]
        extractor = TextFeatureExtractor(max_features=20)
        feats = extractor.fit_transform(corpus)
        self.assertEqual(feats.shape[0], 3)
        self.assertGreater(feats.shape[1], 0)

    def test_03_image_preprocessing(self):
        """Test image resizing and tensor output format."""
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        tensor = load_and_preprocess_image(img)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_04_image_feature_extraction(self):
        """Test ResNet feature vector extraction."""
        img = Image.new("RGB", (224, 224), color=(50, 100, 150))
        extractor = ImageFeatureExtractor()
        feat = extractor.extract_features_single(img)
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape[0], 512)

    def test_05_conflict_detection_logic(self):
        """Test disagreement flag and sarcasm diagnosis."""
        text_res = {"sentiment": "positive", "confidence": 0.85}
        image_res = {"sentiment": "negative", "confidence": 0.80}
        diag = detect_modality_conflict(text_res, image_res)

        self.assertTrue(diag["is_conflict"])
        self.assertEqual(diag["conflict_severity"], "HIGH")
        self.assertIn("Strong Modality Clash", diag["explanation"])

        # Consensus test
        agree_text = {"sentiment": "positive", "confidence": 0.88}
        agree_img = {"sentiment": "positive", "confidence": 0.82}
        diag_agree = detect_modality_conflict(agree_text, agree_img)
        self.assertFalse(diag_agree["is_conflict"])
        self.assertEqual(diag_agree["conflict_severity"], "NONE")

    def test_06_pipeline_inference(self):
        """Test integrated pipeline inference handler."""
        pipeline = IndicSentPipeline()
        dummy_img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        res = pipeline.analyze(text="Test post", image=dummy_img)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["input_modalities"]["has_text"])
        self.assertTrue(res["input_modalities"]["has_image"])


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIndicSentIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
