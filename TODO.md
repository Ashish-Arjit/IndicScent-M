# IndicSent-M: Student Implementation Roadmap (~30% Reserved Work)

This document outlines the designated extension tasks reserved for the five team members.
The core foundation (~70%) is fully functional, runnable, and tested. The tasks below represent advanced modeling, dataset scaling, architectural experiments, and qualitative error analysis to be completed by each student.

---

## Member 1 Tasks (Surya Pratap Singh — Dataset Preparation)

### Task 1.1: Plug in the Full Memotion 7K Dataset
- **Relevant File:** `member1/dataset_preprocessing.py`
- **What needs to be implemented:** Add an automated ingestion function `ingest_memotion_7k(archive_dir)` that parses Memotion's `labels.csv`, matches image filenames to the downloaded image directory, verifies image readability, and maps Memotion's multi-column labels (`overall_sentiment`, `humour`, `sarcasm`, `offensive`, `motivational`) to the standardized 3-class target.
- **Why it matters:** Currently, the pipeline runs on the reproducible 60-sample benchmark dataset. Ingesting the full ~7,000 meme dataset enables large-scale training and statistically significant evaluations.
- **How to test:** Place downloaded Memotion files in `data/raw/memotion/` and run `python member1/dataset_preprocessing.py`.
- **Success Criteria:** `train.csv`, `val.csv`, and `test.csv` contain ~4,900, ~1,050, and ~1,050 verified rows respectively, with 0 missing images in `data_quality_report.json`.

### Task 1.2: Ingest the IndicSentiment Text-Only Track
- **Relevant File:** `member1/dataset_preprocessing.py`
- **What needs to be implemented:** Add a separate ingestion loader for AI4Bharat IndicSentiment (Hindi, Tamil, Telugu, etc.) to store a dedicated Indian-language text evaluation benchmark.
- **Why it matters:** Investigates how well models trained on code-mixed social media generalize to standard Indian-language news and reviews.
- **How to test:** Run `python member1/dataset_preprocessing.py --track indic` and inspect `data/processed/indic_sentiment_clean.csv`.
- **Success Criteria:** A clean CSV containing at least 1,000 Indic text sentences across Hindi/Tamil/Telugu with valid 3-class sentiment labels.

---

## Member 2 Tasks (B.T. Ashish — Text Sentiment Analysis)

### Task 2.1: Pretrained IndicBERT / Multilingual Transformer Feature Extractor
- **Relevant File:** `member2/text_feature_extraction.py`
- **What needs to be implemented:** Implement an `IndicBERTEmbedder` class using Hugging Face `transformers` (`ai4bharat/indic-bert` or `xlm-roberta-base`). Extract mean-pooled contextual embeddings (768 dimensions) from the final hidden layer.
- **Why it matters:** While TF-IDF captures lexical cues and n-grams well, transformer embeddings capture syntax, polysemy, and cross-lingual semantics across Indian scripts.
- **How to test:** Run `python member2/text_feature_extraction.py` with `USE_TRANSFORMER=True` and verify extracted tensor shape is `(N, 768)`.
- **Success Criteria:** Extracted transformer features can be passed directly to `TextSentimentClassifier` without dimensionality or type errors.

### Task 2.2: Transliteration & Romanized Hinglish Normalizer
- **Relevant File:** `member2/text_preprocessing.py`
- **What needs to be implemented:** Expand the Hinglish normalization dictionary for common phonetic spellings (e.g., `bohot`/`bahut`, `khush`/`khus`, `accha`/`acha`/`achha`).
- **Why it matters:** Code-mixed social media exhibits extreme spelling variation that dilutes vocabulary frequency.
- **How to test:** Create test cases with spelling variants in `tests/test_member2_text.py` and run `pytest tests/test_member2_text.py`.
- **Success Criteria:** Canonical words are matched, reducing vocabulary sparsity and improving Macro-F1 on test Hinglish samples.

---

## Member 3 Tasks (CH. Akhil — Image Sentiment Analysis)

### Task 3.1: ResNet-50 / ViT Backbone Comparison
- **Relevant File:** `member3/image_feature_extraction.py`
- **What needs to be implemented:** Add a backbone selection argument (e.g., `backbone='resnet50'` or `backbone='vit_b_16'`) allowing comparison between ResNet-18 (512 dims), ResNet-50 (2048 dims), and Vision Transformer (768 dims).
- **Why it matters:** Vision Transformers frequently capture global semantic meme layouts better than convolutional backbones.
- **How to test:** Run `python member3/image_feature_extraction.py --backbone resnet50` and observe feature vector shape `(2048,)`.
- **Success Criteria:** Feature extraction completes without memory errors and prints the dimension dynamically.

### Task 3.2: Meme Visual Region Masking (Text Bounding Box Removal)
- **Relevant File:** `member3/image_preprocessing.py`
- **What needs to be implemented:** Implement a preprocessing step that detects and masks out burned-in meme text areas (top/bottom banners) so the image classifier focuses purely on visual facial expressions and scene emotion, avoiding textual leakage into the image modality.
- **Why it matters:** Pure visual sentiment should assess image aesthetics, not the text overlay.
- **How to test:** Compare image model performance with and without text masking.
- **Success Criteria:** Masked images maintain high image-only classification accuracy while isolating visual features.

---

## Member 4 Tasks (B. Rushindra Choudary — Multimodal Fusion & Conflict Detection)

### Task 4.1: Late Fusion (Decision-Level Probability Blending)
- **Relevant File:** `member4/feature_fusion.py` and `member4/multimodal_model.py`
- **What needs to be implemented:** Implement late fusion as an alternative to concatenation:
  $P_{\text{final}} = \alpha \cdot P_{\text{text}} + (1 - \alpha) \cdot P_{\text{image}}$
  Add a grid search to find the optimal modality weight $\alpha \in [0.1, 0.9]$ on the validation split.
- **Why it matters:** Compares feature-level (early) fusion against decision-level (late) fusion to determine which is superior for meme sentiment.
- **How to test:** Run `python member4/multimodal_model.py --fusion-method late` and inspect metrics.
- **Success Criteria:** Validation Macro-F1 is printed across different $\alpha$ values and the best $\alpha$ is automatically saved to `models/fusion_scaler.pkl`.

### Task 4.2: Adaptive Conflict Threshold Tuning & Error Analysis
- **Relevant File:** `member4/conflict_detection.py`
- **What needs to be implemented:** Implement an ROC-based threshold optimization function that chooses the optimal confidence differential threshold to maximize F1-score specifically on sarcastic/ironic memes.
- **Why it matters:** Fixed thresholds (0.45) may produce false alarms on ambiguous neutral posts.
- **How to test:** Run `python member4/conflict_detection.py --tune-threshold` and inspect the generated precision-recall curve.
- **Success Criteria:** Outputs a tuned threshold with documented precision and recall in `results/conflict_summary.json`.

---

## Member 5 Tasks (B. Sai Krishna — Integration, Interface & Testing)

### Task 5.1: Batch Evaluation & CSV Upload in Streamlit App
- **Relevant File:** `member5/app.py`
- **What needs to be implemented:** Add a "Batch Processing" sub-tab in `app.py` allowing users to upload a CSV of memes and captions, run predictions across all rows, and download an enriched CSV with predicted sentiments, confidences, and conflict flags.
- **Why it matters:** Enables professors and viva evaluators to test entire datasets with one click during project demonstrations.
- **How to test:** Upload a sample CSV in the Streamlit UI and click "Process Batch".
- **Success Criteria:** A downloadable CSV is returned containing predictions and conflict tags for every uploaded record.

### Task 5.2: Expanded Unit and Regression Testing
- **Relevant File:** `tests/` and `member5/testing.py`
- **What needs to be implemented:** Add test cases for invalid input formats (e.g. grayscale images, corrupted JPEG headers, emojis with no text, empty strings, unsupported languages).
- **Why it matters:** Strengthens system robustness and protects against crashes during live demos.
- **How to test:** Run `pytest tests/` and verify that all test cases pass with 100% success.
- **Success Criteria:** At least 15 comprehensive unit tests passing with zero uncaught exceptions.
