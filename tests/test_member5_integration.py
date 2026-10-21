"""
tests/test_member5_integration.py
Unit tests for Member 5: Unified inference pipeline, input validation, and system status.
"""

import sys
from pathlib import Path
import unittest
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from member5.integration import IndicSentPipeline


class TestMember5Integration(unittest.TestCase):
    """Tests for Member 5 integration pipeline."""

    def test_pipeline_instantiation(self):
        """Verify pipeline instantiates and returns boolean status dict."""
        pipeline = IndicSentPipeline()
        status = pipeline.get_system_status()
        self.assertIn("text_model_loaded", status)
        self.assertIn("image_model_loaded", status)
        self.assertIn("multimodal_model_loaded", status)

    def test_pipeline_empty_input_handling(self):
        """Verify graceful error reporting when neither text nor image is provided."""
        pipeline = IndicSentPipeline()
        res = pipeline.analyze(text="", image=None)
        self.assertEqual(res["status"], "error")
        self.assertIn("at least text or an image", res["message"])

    def test_pipeline_valid_text_only_structure(self):
        """Verify output dictionary structure for text-only input."""
        pipeline = IndicSentPipeline()
        res = pipeline.analyze(text="Hello world test", image=None)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["input_modalities"]["has_text"])
        self.assertFalse(res["input_modalities"]["has_image"])
        self.assertIsNotNone(res["text_prediction"])
        self.assertIsNone(res["image_prediction"])
        self.assertIsNone(res["conflict_diagnosis"])


if __name__ == "__main__":
    unittest.main()
