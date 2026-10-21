# Member 1: Dataset Preparation & Quality Assurance

**Owner:** Surya Pratap Singh  
**Milestone:** Dataset Engineering, Cleaning, Label Standardization, and Data Health Reporting

---

## Responsibilities

1. **Dataset Inspection:** Load, inspect column structures, identify data types, missing rates, and verify paired samples (`dataset_inspection.py`).
2. **Dataset Preprocessing:** Standardize raw class labels into `negative`, `neutral`, `positive`. Remove corrupted or unreadable text-image pairs and prevent duplicate text leakage (`dataset_preprocessing.py`).
3. **Reproducible Splitting:** Generate stratified 70% Train, 15% Validation, and 15% Test splits using a fixed random seed (`RANDOM_SEED=42`).
4. **Label Analysis:** Track class frequencies and percentages to monitor dataset balance (`label_analysis.py`).
5. **Quality Reporting:** Export detailed JSON and Markdown reports summarizing modality completeness and dataset integrity (`data_quality_report.py`).

---

## Exposed Functions for the Team

- `load_or_create_dataset()`: Returns `(train_df, val_df, test_df)`.
- `clean_and_split_dataset(df)`: Cleans arbitrary raw CSV and creates stratified splits.
- `inspect_dataframe(df, name)`: Inspects column types and missingness.
- `validate_image_text_pairs(df)`: Validates image file readability and text presence.
- `generate_data_quality_report()`: Generates report in `results/`.

---

## Usage

```bash
# Run standalone dataset preparation
python member1/dataset_preprocessing.py

# Inspect label distributions
python member1/label_analysis.py

# Generate data quality report
python member1/data_quality_report.py
```
