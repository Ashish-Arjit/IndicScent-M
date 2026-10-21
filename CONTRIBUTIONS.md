# IndicSent-M: Team Contributions & Module Ownership

This document defines the architectural division of labor and component ownership for the five members of the **IndicSent-M** project at Bennett University.

---

## Team Overview

| Member | Name | Designated Folder | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Member 1** | Surya Pratap Singh | `member1/` | Dataset preparation, inspection, cleaning, label mapping, and data quality reporting |
| **Member 2** | B.T. Ashish | `member2/` | Code-mixed text preprocessing, emoji preservation, TF-IDF / IndicBERT feature extraction, and text classifier |
| **Member 3** | CH. Akhil | `member3/` | Image tensor transforms, pretrained ResNet visual embedding extraction, and image classifier |
| **Member 4** | B. Rushindra Choudary | `member4/` | Multimodal feature fusion, multimodal classifier, conflict/sarcasm detection, and 3-way model comparison |
| **Member 5** | B. Sai Krishna | `member5/` | Unified inference coordinator, Streamlit interactive UI, CLI results dashboard, and integration test suite |

---

## Detailed Breakdown by Member

### Member 1: Surya Pratap Singh — Dataset Preparation & Quality Assurance
- **Folder:** `member1/`
- **Implemented Files:**
  - `dataset_inspection.py`: Inspects column types, schema consistency, missing rates, and validates paired image-text file existence.
  - `dataset_preprocessing.py`: Cleans raw metadata, standardizes labels into 3 classes (`negative`, `neutral`, `positive`), handles missing values, prevents duplicate text leakage, generates reproducible sample benchmark memes with PIL, and splits data into stratified Train (70%), Val (15%), and Test (15%) splits.
  - `label_analysis.py`: Computes sentiment class counts and percentage balances across each split.
  - `data_quality_report.py`: Generates comprehensive JSON and Markdown health summaries.
  - `README.md`: Technical documentation for dataset utilities.
- **Dependencies:** `pandas`, `numpy`, `pillow`, `scikit-learn`
- **Key Functions Exposed:**
  - `load_or_create_dataset()`
  - `clean_and_split_dataset(df)`
  - `inspect_dataframe(df, name)`
  - `validate_image_text_pairs(df)`
  - `generate_data_quality_report()`
- **Reserved Student Tasks:** Plug in the full 7,000 Memotion dataset archive, download AI4Bharat IndicSentiment track, and inspect code-mixed Dravidian language tags.

---

### Member 2: B.T. Ashish — Text Sentiment Analysis
- **Folder:** `member2/`
- **Implemented Files:**
  - `text_preprocessing.py`: Normalizes social media text, removes URLs/handles, retains Indic Unicode scripts (Devanagari, Bengali, Tamil, Telugu), compresses character repetitions, and maps emoticons/emojis to sentiment tokens.
  - `text_feature_extraction.py`: N-gram TF-IDF vectorizer (unigrams + bigrams with sublinear scaling) optimized for code-mixed / Hinglish text. Saves fitted vectorizer to `models/text_vectorizer.pkl`.
  - `text_model.py`: Trainable class-balanced Logistic Regression sentiment model with calibrated probability outputs. Saves to `models/text_model.pkl`.
  - `evaluate_text.py`: Evaluates text model on `data/processed/test.csv`, exports `results/text_metrics.json` and `results/text_predictions.csv`.
  - `README.md`: Module documentation and usage.
- **Dependencies:** `scikit-learn`, `regex`, `pandas`, `numpy`
- **Key Functions & Classes Exposed:**
  - `clean_text(text: str) -> str`
  - `TextFeatureExtractor`
  - `TextSentimentClassifier`
  - `train_text_pipeline()`
  - `evaluate_text_model()`
- **Reserved Student Tasks:** Benchmark fine-tuning an `IndicBERT` or `XLM-RoBERTa` Transformer encoder against the TF-IDF baseline and analyze Hindi/Hinglish vocabulary coverage.

---

### Member 3: CH. Akhil — Image Sentiment Analysis
- **Folder:** `member3/`
- **Implemented Files:**
  - `image_preprocessing.py`: Image loading from disk or PIL, RGB conversion, aspect resizing (224x224), and torchvision ImageNet tensor normalization.
  - `image_feature_extraction.py`: Pretrained ResNet-18 backbone with classification head replaced by Identity to extract dense 512-dimensional visual embedding vectors with frozen weights in evaluation mode.
  - `image_model.py`: Image-only sentiment classifier head trained on extracted visual representations. Saves to `models/image_model.pkl`.
  - `evaluate_image.py`: Evaluates image model on test memes, exports `results/image_metrics.json` and `results/image_predictions.csv`.
  - `README.md`: Module documentation and usage.
- **Dependencies:** `torch`, `torchvision`, `pillow`, `scikit-learn`, `numpy`
- **Key Functions & Classes Exposed:**
  - `load_and_preprocess_image(image_input) -> torch.Tensor`
  - `ImageFeatureExtractor`
  - `ImageSentimentClassifier`
  - `train_image_pipeline()`
  - `evaluate_image_model()`
- **Reserved Student Tasks:** Experiment with deeper backbones (ResNet-50, ViT) and visual meme OCR region masking.

---

### Member 4: B. Rushindra Choudary — Multimodal Fusion & Conflict Detection
- **Folder:** `member4/`
- **Implemented Files:**
  - `feature_fusion.py`: Early feature fusion concatenating text features and visual features into a unified vector with `StandardScaler` normalization.
  - `multimodal_model.py`: Multimodal sentiment classifier learning joint representations from text and images. Saves to `models/multimodal_model.pkl`.
  - `conflict_detection.py`: Diagnostic conflict and sarcasm detection engine. Compares text vs image predictions and confidence differentials. Flags high-severity clashes and exports `results/conflict_analysis.csv` and `results/conflict_summary.json`.
  - `compare_models.py`: Compares Text-only vs Image-only vs Multimodal side-by-side across Accuracy, Macro-F1, Precision, and Recall on the identical test split.
  - `README.md`: Module documentation and usage.
- **Dependencies:** `scikit-learn`, `numpy`, `pandas`, `torch`, `torchvision`
- **Key Functions & Classes Exposed:**
  - `MultimodalFeatureFusion`
  - `MultimodalSentimentClassifier`
  - `detect_modality_conflict(text_res, img_res, mm_res) -> Dict`
  - `analyze_test_set_conflicts()`
  - `compare_all_models()`
- **Reserved Student Tasks:** Implement late fusion (weighted probability blending), cross-attention fusion, and tune conflict confidence margins.

---

### Member 5: B. Sai Krishna — Integration, Interface & Testing
- **Folder:** `member5/`
- **Implemented Files:**
  - `integration.py`: End-to-end `IndicSentPipeline` coordinating text, image, and multimodal models into a single callable API with system health checks.
  - `app.py`: Interactive Streamlit application featuring live multimodal inference, real-time conflict badge alerts, 3-way benchmark comparison dashboard, confusion matrices, and test-set conflict explorer.
  - `results_dashboard.py`: CLI console dashboard summarizing dataset health, model benchmarks, and conflict statistics.
  - `testing.py`: Integration test suite validating preprocessing, feature shapes, model inference, and conflict detection.
  - `README.md`: Module documentation and usage.
- **Dependencies:** `streamlit`, `pandas`, `pillow`, `matplotlib`, `pytest`
- **Key Functions & Classes Exposed:**
  - `IndicSentPipeline`
  - `print_project_dashboard()`
  - `run_tests()`
- **Reserved Student Tasks:** Add batch CSV file upload in Streamlit UI, export downloadable PDF/CSV reports, and expand unit test coverage.

---

## Recommended Student Workflow
1. Each member clones the repository and creates their own feature branch (`feature/memberX-<name>`).
2. Run `python main.py --run-tests` to ensure the baseline pipeline passes.
3. Review `TODO.md` for designated Member-specific extension tasks.
4. Test and commit code strictly within your designated `memberX/` directory.
