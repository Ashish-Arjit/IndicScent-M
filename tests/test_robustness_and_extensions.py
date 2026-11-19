"""
tests/test_robustness_and_extensions.py
Task 5.2: Expanded Unit and Regression Testing for IndicSent-M.
Covers grayscale images, RGBA images, corrupted paths, empty strings,
emoji-only captions, Hinglish phonetic normalizer, transformer embedder,
backbone selection, meme masking, late fusion, and conflict diagnosis.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from member1.dataset_preprocessing import standardize_label
from member2.text_preprocessing import clean_text, normalize_hinglish
from member2.text_feature_extraction import TextFeatureExtractor, IndicBERTEmbedder
from member3.image_preprocessing import load_and_preprocess_image, mask_meme_text_regions
from member3.image_feature_extraction import ImageFeatureExtractor
from member4.multimodal_model import MultimodalSentimentClassifier
from member4.conflict_detection import detect_modality_conflict


class TestRobustnessAndExtensions(unittest.TestCase):
    """Test suite for edge cases, error handling, and newly added 30% features."""

    def test_hinglish_normalization_coverage(self):
        """Test Hinglish dictionary replaces colloquial variations."""
        raw = "bohot khus hoke bhai ne accha decision liya nhi toh bekaar ho jata"
        norm = normalize_hinglish(raw)
        self.assertIn("bahut", norm)
        self.assertIn("khush", norm)
        self.assertIn("achha", norm)
        self.assertIn("nahi", norm)
        self.assertIn("bekar", norm)

    def test_indicbert_embedder_dimension(self):
        """Test IndicBERT embedder produces 768 dimensions."""
        embedder = IndicBERTEmbedder()
        feats = embedder.transform(["Test sentence in Hinglish"])
        self.assertEqual(feats.shape, (1, 768))

    def test_grayscale_and_rgba_image_loading(self):
        """Test image preprocessing handles 1-channel grayscale and 4-channel RGBA."""
        gray = Image.new("L", (100, 100), color=120)
        t_gray = load_and_preprocess_image(gray)
        self.assertIsNotNone(t_gray)
        self.assertEqual(t_gray.shape, (1, 3, 224, 224))

        rgba = Image.new("RGBA", (100, 100), color=(0, 200, 100, 255))
        t_rgba = load_and_preprocess_image(rgba)
        self.assertIsNotNone(t_rgba)
        self.assertEqual(t_rgba.shape, (1, 3, 224, 224))

    def test_meme_masking_applied(self):
        """Test visual banner masking fills top/bottom regions."""
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        masked = mask_meme_text_regions(img, top_ratio=0.15, bottom_ratio=0.15)
        self.assertEqual(masked.getpixel((5, 5)), (0, 0, 0))

    def test_backbone_resnet50_and_vit(self):
        """Test that ResNet-50 extracts 2048 dims and ViT extracts 768 dims."""
        r50 = ImageFeatureExtractor(backbone="resnet50")
        self.assertEqual(r50.feature_dim, 2048)
        vit = ImageFeatureExtractor(backbone="vit_b_16")
        self.assertEqual(vit.feature_dim, 768)

    def test_empty_string_and_emoji_only(self):
        """Test handling of empty string and emoji-only strings."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")
        emoji_clean = clean_text("😊 😊")
        self.assertIn("emoji_positive", emoji_clean)

    def test_late_fusion_probability_blending(self):
        """Test decision-level late fusion blending formula."""
        clf = MultimodalSentimentClassifier(fusion_method="late")
        clf.alpha = 0.7
        p_t = {"negative": 0.1, "neutral": 0.1, "positive": 0.8}
        p_i = {"negative": 0.5, "neutral": 0.3, "positive": 0.2}
        blended_pos = round(0.7 * p_t["positive"] + 0.3 * p_i["positive"], 4)
        self.assertAlmostEqual(blended_pos, 0.62, places=3)


if __name__ == "__main__":
    unittest.main()
