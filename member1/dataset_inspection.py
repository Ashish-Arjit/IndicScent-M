"""
member1/dataset_inspection.py
Responsible for inspecting dataset structures, column definitions, data types,
missing values, and verifying image-text pairings.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def inspect_dataframe(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """Inspect columns, shape, missing values, and data types of a DataFrame."""
    inspection = {
        "dataset_name": name,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_percentages": {
            col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns
        },
        "duplicate_rows": int(df.duplicated().sum())
    }
    return inspection


def validate_image_text_pairs(
    df: pd.DataFrame,
    image_col: str = "image_path",
    text_col: str = "text"
) -> Dict[str, Any]:
    """
    Validate that each row has non-empty text and a readable image on disk.
    Returns status counts and invalid sample IDs.
    """
    valid_pairs = 0
    missing_images = []
    corrupt_images = []
    missing_texts = []

    for idx, row in df.iterrows():
        sample_id = row.get("sample_id", idx)
        
        # Check text
        text_val = str(row.get(text_col, "")).strip()
        if not text_val or pd.isna(row.get(text_col)):
            missing_texts.append(sample_id)

        # Check image
        img_path = Path(str(row.get(image_col, "")))
        if not img_path.exists():
            missing_images.append(sample_id)
        else:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_pairs += 1
            except Exception:
                corrupt_images.append(sample_id)

    return {
        "total_checked": len(df),
        "valid_pairs": valid_pairs,
        "missing_images_count": len(missing_images),
        "corrupt_images_count": len(corrupt_images),
        "missing_texts_count": len(missing_texts),
        "missing_image_ids": missing_images[:10],
        "corrupt_image_ids": corrupt_images[:10],
        "missing_text_ids": missing_texts[:10],
    }


def print_inspection_summary(inspection: Dict[str, Any]) -> None:
    """Print readable summary of the dataset inspection."""
    print(f"\n--- Inspection: {inspection.get('dataset_name', 'Dataset')} ---")
    print(f"Total Rows: {inspection['total_rows']}, Columns: {inspection['total_columns']}")
    print("Columns and Missing Values:")
    for col in inspection["columns"]:
        miss = inspection["missing_counts"][col]
        pct = inspection["missing_percentages"][col]
        dtype = inspection["dtypes"][col]
        print(f"  - {col} ({dtype}): {miss} missing ({pct}%)")
    print(f"Duplicate Rows: {inspection['duplicate_rows']}")


if __name__ == "__main__":
    # Test on processed data if available
    test_csv = PROCESSED_DATA_DIR / "train.csv"
    if test_csv.exists():
        df_sample = pd.read_csv(test_csv)
        summary = inspect_dataframe(df_sample, "Train Split")
        print_inspection_summary(summary)
    else:
        print("No processed train.csv found to inspect. Run dataset_preprocessing.py first.")
