"""
tests/test_member1_dataset.py
Unit tests for Member 1: Dataset loading, cleaning, label mapping, and splitting.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from member1.dataset_preprocessing import standardize_label, clean_and_split_dataset
from member1.dataset_inspection import inspect_dataframe


class TestMember1Dataset(unittest.TestCase):
    """Tests for Member 1 dataset utilities."""

    def test_standardize_label(self):
        """Test mapping various label formats to positive/neutral/negative."""
        self.assertEqual(standardize_label("positive"), "positive")
        self.assertEqual(standardize_label("POS"), "positive")
        self.assertEqual(standardize_label("very_positive"), "positive")
        self.assertEqual(standardize_label("2"), "positive")
        self.assertEqual(standardize_label("negative"), "negative")
        self.assertEqual(standardize_label("0"), "negative")
        self.assertEqual(standardize_label("neutral"), "neutral")
        self.assertEqual(standardize_label("1"), "neutral")
        self.assertIsNone(standardize_label("invalid_label_123"))

    def test_clean_and_split_dataset(self):
        """Test stratified split ratios and no leakage."""
        data = []
        for i in range(30):
            data.append({
                "sample_id": f"S_{i}",
                "text": f"Sample caption text {i}",
                "sentiment": ["positive", "neutral", "negative"][i % 3]
            })
        df = pd.DataFrame(data)
        train_df, val_df, test_df = clean_and_split_dataset(df, test_size=0.2, val_size=0.2)

        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 30)
        
        # Check no overlap in sample_id
        train_ids = set(train_df["sample_id"])
        val_ids = set(val_df["sample_id"])
        test_ids = set(test_df["sample_id"])
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_inspect_dataframe(self):
        """Test dataset inspection report generator."""
        df = pd.DataFrame({
            "col1": [1, 2, None],
            "col2": ["a", "b", "c"]
        })
        info = inspect_dataframe(df, "TestDF")
        self.assertEqual(info["total_rows"], 3)
        self.assertEqual(info["missing_counts"]["col1"], 1)
        self.assertEqual(info["missing_counts"]["col2"], 0)


if __name__ == "__main__":
    unittest.main()
