"""
tests/test_member4_fusion.py
Unit tests for Member 4: Multimodal feature concatenation, conflict detection, and comparison.
"""

import sys
from pathlib import Path
import unittest
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from member4.feature_fusion import MultimodalFeatureFusion
from member4.conflict_detection import detect_modality_conflict


class TestMember4Fusion(unittest.TestCase):
    """Tests for Member 4 multimodal module."""

    def test_feature_fusion_concatenation(self):
        """Test that fused features concatenate text (e.g. 5000) and image (512) vectors."""
        texts = ["Positive post", "Negative report", "Neutral update"]
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        fusion = MultimodalFeatureFusion()
        fusion.fit(texts, imgs)
        
        vec = fusion.extract_single("Positive post", imgs[0])
        # Text feature dim + image feature dim (512)
        self.assertGreater(vec.shape[1], 512)

    def test_conflict_detection_severities(self):
        """Test high, moderate, and zero conflict classifications."""
        # 1. High conflict (both high confidence but disagree)
        t_high = {"sentiment": "positive", "confidence": 0.85}
        i_high = {"sentiment": "negative", "confidence": 0.82}
        res_high = detect_modality_conflict(t_high, i_high)
        self.assertTrue(res_high["is_conflict"])
        self.assertEqual(res_high["conflict_severity"], "HIGH")

        # 2. Moderate conflict (low confidence disagreement)
        t_low = {"sentiment": "positive", "confidence": 0.40}
        i_low = {"sentiment": "neutral", "confidence": 0.38}
        res_low = detect_modality_conflict(t_low, i_low)
        self.assertTrue(res_low["is_conflict"])
        self.assertEqual(res_low["conflict_severity"], "MODERATE")

        # 3. Consensus (no conflict)
        t_agree = {"sentiment": "positive", "confidence": 0.90}
        i_agree = {"sentiment": "positive", "confidence": 0.88}
        res_agree = detect_modality_conflict(t_agree, i_agree)
        self.assertFalse(res_agree["is_conflict"])
        self.assertEqual(res_agree["conflict_severity"], "NONE")


if __name__ == "__main__":
    unittest.main()
