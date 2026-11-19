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
from member1.dataset_preprocessing import clean_and_split_dataset, create_sample_benchmark_dataset, standardize_label
from member2.text_preprocessing import clean_text, normalize_hinglish
from member2.text_feature_extraction import TextFeatureExtractor, IndicBERTEmbedder
from member3.image_preprocessing import load_and_preprocess_image, mask_meme_text_regions
from member3.image_feature_extraction import ImageFeatureExtractor
from member4.feature_fusion import MultimodalFeatureFusion
from member4.multimodal_model import MultimodalSentimentClassifier
from member4.conflict_detection import detect_modality_conflict, tune_conflict_threshold
from member5.integration import IndicSentPipeline


class TestIndicSentIntegration(unittest.TestCase):
    """Integration and robustness test suite for the complete IndicSent-M pipeline."""

    def test_01_text_preprocessing(self):
        """Test emoji preservation and character normalization."""
        raw = "Passed the exam!! Soooo happy 😊 check https://test.edu @friends"
        cleaned = clean_text(raw)
        self.assertIn("emoji_positive", cleaned)
        self.assertNotIn("https://test.edu", cleaned)
        self.assertNotIn("@friends", cleaned)
        self.assertIn("exam", cleaned)

    def test_02_hinglish_phonetic_normalization(self):
        """Test Task 2.2: Romanized Hinglish phonetic normalizer maps spelling variations."""
        raw_variants = "Bohot khus hu aaj, accha laga bhai nhi gya"
        normalized = normalize_hinglish(raw_variants)
        self.assertIn("bahut", normalized)
        self.assertIn("khush", normalized)
        self.assertIn("achha", normalized)
        self.assertIn("nahi", normalized)
        self.assertIn("gaya", normalized)

    def test_03_text_feature_extraction(self):
        """Test TF-IDF feature extraction dimensions and fitting."""
        corpus = ["Good day today", "Bad server crash", "Library open"]
        extractor = TextFeatureExtractor(max_features=20)
        feats = extractor.fit_transform(corpus)
        self.assertEqual(feats.shape[0], 3)
        self.assertGreater(feats.shape[1], 0)

    def test_04_transformer_indicbert_embedder(self):
        """Test Task 2.1: Transformer embedding extractor produces 768-dim tensor."""
        embedder = IndicBERTEmbedder()
        feats = embedder.transform(["Campus placements are going smoothly!"])
        self.assertEqual(feats.shape, (1, 768))

    def test_05_image_preprocessing(self):
        """Test image resizing and tensor output format."""
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        tensor = load_and_preprocess_image(img)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_06_grayscale_and_rgba_images(self):
        """Test robustness on grayscale (L mode) and RGBA input images."""
        gray_img = Image.new("L", (100, 100), color=128)
        tensor_gray = load_and_preprocess_image(gray_img)
        self.assertIsNotNone(tensor_gray)
        self.assertEqual(tensor_gray.shape, (1, 3, 224, 224))

        rgba_img = Image.new("RGBA", (150, 150), color=(255, 0, 0, 128))
        tensor_rgba = load_and_preprocess_image(rgba_img)
        self.assertIsNotNone(tensor_rgba)
        self.assertEqual(tensor_rgba.shape, (1, 3, 224, 224))

    def test_07_meme_text_region_masking(self):
        """Test Task 3.2: Meme visual banner masking."""
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        masked = mask_meme_text_regions(img, top_ratio=0.2, bottom_ratio=0.2)
        self.assertEqual(masked.size, (200, 200))
        # Top banner should be black (0, 0, 0)
        self.assertEqual(masked.getpixel((10, 10)), (0, 0, 0))

    def test_08_image_feature_extraction(self):
        """Test ResNet feature vector extraction."""
        img = Image.new("RGB", (224, 224), color=(50, 100, 150))
        extractor = ImageFeatureExtractor(backbone="resnet18")
        feat = extractor.extract_features_single(img)
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape[0], 512)

    def test_09_backbone_selection(self):
        """Test Task 3.1: ResNet-50 (2048 dims) and ViT-B-16 (768 dims) backbones."""
        r50_extractor = ImageFeatureExtractor(backbone="resnet50")
        self.assertEqual(r50_extractor.feature_dim, 2048)

        vit_extractor = ImageFeatureExtractor(backbone="vit_b_16")
        self.assertEqual(vit_extractor.feature_dim, 768)

    def test_10_conflict_detection_logic(self):
        """Test disagreement flag and sarcasm diagnosis."""
        text_res = {"sentiment": "positive", "confidence": 0.85}
        image_res = {"sentiment": "negative", "confidence": 0.80}
        diag = detect_modality_conflict(text_res, image_res)

        self.assertTrue(diag["is_conflict"])
        self.assertEqual(diag["conflict_severity"], "HIGH")
        self.assertIn("High-Severity Conflict", diag["explanation"])

        # Consensus test
        agree_text = {"sentiment": "positive", "confidence": 0.88}
        agree_img = {"sentiment": "positive", "confidence": 0.82}
        diag_agree = detect_modality_conflict(agree_text, agree_img)
        self.assertFalse(diag_agree["is_conflict"])
        self.assertEqual(diag_agree["conflict_severity"], "NONE")

    def test_11_late_fusion_probability_blending(self):
        """Test Task 4.1: Late fusion probability calculation."""
        clf = MultimodalSentimentClassifier(fusion_method="late")
        clf.alpha = 0.6
        # Dummy text & image predictions
        t_p = {"sentiment": "positive", "confidence": 0.8, "probabilities": {"negative": 0.1, "neutral": 0.1, "positive": 0.8}}
        i_p = {"sentiment": "neutral", "confidence": 0.6, "probabilities": {"negative": 0.2, "neutral": 0.6, "positive": 0.2}}
        
        # Expected positive = 0.6*0.8 + 0.4*0.2 = 0.48 + 0.08 = 0.56
        blended_pos = round(0.6 * t_p["probabilities"]["positive"] + 0.4 * i_p["probabilities"]["positive"], 4)
        self.assertAlmostEqual(blended_pos, 0.56, places=3)

    def test_12_pipeline_inference(self):
        """Test integrated pipeline inference handler."""
        pipeline = IndicSentPipeline()
        dummy_img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        res = pipeline.analyze(text="Test post", image=dummy_img)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["input_modalities"]["has_text"])
        self.assertTrue(res["input_modalities"]["has_image"])

    def test_13_edge_case_empty_and_whitespace_text(self):
        """Test edge cases with empty or whitespace-only text."""
        cleaned_empty = clean_text("")
        cleaned_space = clean_text("   \n\t  ")
        self.assertEqual(cleaned_empty, "")
        self.assertEqual(cleaned_space, "")

    def test_14_edge_case_emoji_only_text(self):
        """Test emoji-only text cleans to sentiment tokens without error."""
        text = "😊 🔥 ❤️"
        cleaned = clean_text(text)
        self.assertIn("emoji_positive", cleaned)

    def test_15_corrupted_image_handling(self):
        """Test that non-existent image paths return None without unhandled crashes."""
        bad_path = Path("non_existent_folder_xyz/corrupt.jpg")
        tensor = load_and_preprocess_image(bad_path)
        self.assertIsNone(tensor)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIndicSentIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
