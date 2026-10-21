# Member 4: Multimodal Fusion & Conflict Detection

**Owner:** B. Rushindra Choudary  
**Milestone:** Feature Fusion, Multimodal Classifier, Disagreement Diagnosis, and Comparative Evaluation

---

## Responsibilities

1. **Feature Fusion:** Concatenate text feature representations (TF-IDF/IndicBERT) and image feature representations (ResNet-18) into a unified embedding with standard normalization (`feature_fusion.py`).
2. **Multimodal Model:** Train and calibrate a unified sentiment classifier that learns from both text captions and visual imagery (`multimodal_model.py`).
3. **Conflict Detection:** Implement a diagnostic pipeline that detects when text-only and image-only models disagree, measures confidence differentials, and identifies meme sarcasm/irony (`conflict_detection.py`).
4. **Comparative Analysis:** Collate genuine test-set evaluation metrics across all three modalities and export side-by-side performance tables (`compare_models.py`).

---

## Exposed Functions & Classes

- `MultimodalFeatureFusion`: Concatenates text and visual embeddings and fits a scaler.
- `MultimodalSentimentClassifier`: Full train/predict/evaluate pipeline for multimodal data.
- `detect_modality_conflict(text_res, img_res, mm_res) -> Dict`: Single sample disagreement detector.
- `analyze_test_set_conflicts() -> Dict`: Batch conflict analysis across the test split.
- `compare_all_models() -> pd.DataFrame`: Aggregates real metrics into `results/model_comparison.csv`.

---

## Usage

```bash
# Train multimodal model
python member4/multimodal_model.py

# Run conflict detection on test set
python member4/conflict_detection.py

# Generate 3-way comparative benchmark table
python member4/compare_models.py
```
