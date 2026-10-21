# IndicSent-M: Multimodal Sentiment Analysis for Indian Languages Using Text and Images

**Bennett University — Department of Computer Science & Engineering (AI & ML)**  
**Course Project: Natural Language Processing**

---

## 1. Project Overview

Social media opinions are rarely expressed through text alone. Modern internet culture in India is intensely multimodal—combining captions, Romanized/code-mixed slang (Hinglish), regional Indian scripts (Hindi, Tamil, Telugu), emojis, and viral meme images. 

Standard text-only sentiment classifiers often fail when confronted with **mixed signals**, such as a smiling face coupled with a sarcastic caption (*"Great job guys, broken code pushed straight to production on Friday!"*). Conversely, image-only models miss linguistic nuance and punchlines.

**IndicSent-M** implements, benchmarks, and contrasts three fundamental paradigms:
1. **Text-Only Sentiment Analysis:** Character and word n-gram TF-IDF representations / IndicBERT embeddings with calibrated classification.
2. **Image-Only Sentiment Analysis:** Deep visual feature representations extracted via a pretrained ResNet-18 convolutional backbone.
3. **Multimodal Fusion:** Early feature concatenation and normalization combining textual semantics and visual imagery.
4. **Confidence-Aware Conflict Detection:** A diagnostic engine that compares unimodal predictions, flags disagreements, computes confidence differentials, and identifies meme irony/sarcasm.

---

## 2. Team Members & Roles

| Member | Name | Designated Module | Area of Ownership |
| :--- | :--- | :--- | :--- |
| **Member 1** | Surya Pratap Singh | `member1/` | Dataset preparation, pair validation, label standardization, stratified splitting, and data quality reporting |
| **Member 2** | B.T. Ashish | `member2/` | Multilingual text cleaning, emoji mapping, TF-IDF vectorization, and text sentiment classification |
| **Member 3** | CH. Akhil | `member3/` | Image preprocessing, ResNet-18 visual feature extraction, and image sentiment classification |
| **Member 4** | B. Rushindra Choudary | `member4/` | Multimodal feature fusion, multimodal classifier, conflict detection engine, and 3-way comparative evaluation |
| **Member 5** | B. Sai Krishna | `member5/` | Unified inference coordinator, interactive Streamlit UI, results dashboard, and integration test suite |

---

## 3. Project Directory Structure

```text
IndicSent-M/
│
├── member1/                     # Member 1: Dataset Preparation (Surya Pratap Singh)
│   ├── dataset_inspection.py    # Column inspection, types, missing rates, pair integrity
│   ├── dataset_preprocessing.py # Cleaning, label mapping, reproducible sample generator, stratified splitting
│   ├── label_analysis.py        # Split label frequency and balance analysis
│   ├── data_quality_report.py   # Comprehensive Markdown & JSON data health report
│   └── README.md
│
├── member2/                     # Member 2: Text Sentiment Analysis (B.T. Ashish)
│   ├── text_preprocessing.py    # URL/handle stripping, emoji preservation, Indic normalization
│   ├── text_feature_extraction.py # N-gram TF-IDF vectorizer (saves to models/text_vectorizer.pkl)
│   ├── text_model.py            # Calibrated Logistic Regression classifier (saves to models/text_model.pkl)
│   ├── evaluate_text.py         # Evaluates on test split, exports metrics & predictions
│   └── README.md
│
├── member3/                     # Member 3: Image Sentiment Analysis (CH. Akhil)
│   ├── image_preprocessing.py   # Resizing (224x224), RGB conversion, ImageNet tensor normalization
│   ├── image_feature_extraction.py # Pretrained ResNet-18 (512-dim visual embeddings)
│   ├── image_model.py           # Image sentiment classifier (saves to models/image_model.pkl)
│   ├── evaluate_image.py        # Evaluates on test memes, exports metrics & predictions
│   └── README.md
│
├── member4/                     # Member 4: Multimodal Fusion & Conflict Detection (B. Rushindra Choudary)
│   ├── feature_fusion.py        # Feature concatenation & StandardScaler normalization
│   ├── multimodal_model.py      # Unified multimodal classifier (saves to models/multimodal_model.pkl)
│   ├── conflict_detection.py    # Disagreement & sarcasm diagnostic pipeline
│   ├── compare_models.py        # Side-by-side benchmark table comparing all 3 modalities
│   └── README.md
│
├── member5/                     # Member 5: Integration & Application (B. Sai Krishna)
│   ├── app.py                   # Streamlit web application with live demo, benchmarks & conflict viewer
│   ├── integration.py           # End-to-end inference coordinator (IndicSentPipeline)
│   ├── results_dashboard.py     # Terminal console summary dashboard
│   ├── testing.py               # Integration test runner
│   └── README.md
│
├── data/                        # Datasets and metadata
│   ├── raw/                     # Original CSVs and archives
│   ├── processed/               # Stratified clean splits (train.csv, val.csv, test.csv)
│   ├── sample_memes/            # Paired sample meme images for instant zero-dependency testing
│   └── README.md
│
├── models/                      # Serialized weights and vectorizers
│   └── README.md
│
├── results/                     # Real evaluation metrics and prediction CSVs
│   └── README.md
│
├── tests/                       # Automated pytest and unittest test suite
│   ├── __init__.py
│   ├── test_member1_dataset.py
│   ├── test_member2_text.py
│   ├── test_member3_image.py
│   ├── test_member4_fusion.py
│   └── test_member5_integration.py
│
├── config.py                    # Central paths, label dictionaries, and hyperparameters
├── main.py                      # Master command-line orchestrator
├── requirements.txt             # Project dependencies
├── CONTRIBUTIONS.md             # Detailed breakdown of individual student ownership
├── TODO.md                      # Remaining ~30% tasks reserved for student completion
└── README.md                    # Master documentation
```

---

## 4. Setup & Installation

### Step 1: Clone or Navigate to the Repository
```bash
cd IndicSent-M
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Execution Workflow

The project is designed with a unified CLI orchestrator (`main.py`):

### Option A: Run Complete Pipeline End-to-End
Runs data preparation, trains all 3 models, evaluates performance, analyzes conflicts, and prints the summary dashboard:
```bash
python main.py --run-all
```

### Option B: Run Individual Modules Step-by-Step
```bash
# 1. Member 1: Dataset preparation & quality report
python main.py --prepare-data

# 2. Member 2: Train text-only sentiment classifier
python main.py --train-text

# 3. Member 3: Train image-only sentiment classifier
python main.py --train-image

# 4. Member 4: Train multimodal fusion classifier
python main.py --train-multimodal

# 5. Evaluate all models and diagnose conflicts
python main.py --evaluate-all
```

### Option C: Launch the Interactive Streamlit Web App
```bash
streamlit run member5/app.py
```
*Or via CLI flag:*
```bash
python main.py --run-app
```

---

## 6. Running Tests

Run the full integration test suite:
```bash
python main.py --run-tests
```
*Or using pytest:*
```bash
pytest tests/ -v
```

---

## 7. Conflict Detection Feature

The project includes an intelligent **modality disagreement diagnosis**:
1. Obtains unimodal text prediction and confidence ($c_{\text{text}}$).
2. Obtains unimodal image prediction and confidence ($c_{\text{image}}$).
3. If predictions differ:
   - **High Severity Clash:** Both modalities have confidence $\ge 0.45$, indicating intentional sarcasm, mixed signals, or ironic humor.
   - **Moderate Disagreement:** One modality is uncertain.
4. Compares the disagreement with the multimodal model's decision to determine whether the fusion model synthesized a resolution or prioritized text/image.

---

## 8. Implementation Status

- **Completed Core (~70%):**
  - Full modular 5-member folder architecture.
  - Complete dataset loading, cleaning, validation, and stratified splitting.
  - Reproducible offline benchmark dataset generator (60 paired memes + captions).
  - Code-mixed text cleaning, emoji preservation, and TF-IDF feature extraction.
  - Pretrained ResNet-18 visual feature extraction and image classification head.
  - Early feature fusion pipeline with StandardScaler normalization.
  - Multimodal classifier training and evaluation.
  - Confidence-aware conflict and sarcasm detection engine.
  - 3-way comparative evaluation table (Accuracy, Macro-F1, Precision, Recall, Confusion Matrix).
  - Interactive multi-tab Streamlit web application.
  - Complete integration and unit test suite across all 5 modules.
- **Reserved for Students (~30%):**
  - Ingesting full 7K Memotion dataset archive and IndicSentiment track (`member1`).
  - Pretrained IndicBERT / XLM-R Transformer feature extraction (`member2`).
  - ResNet-50 / ViT comparison and meme text masking (`member3`).
  - Late fusion probability blending and ROC-based conflict threshold tuning (`member4`).
  - Batch CSV processing in Streamlit UI and expanded regression tests (`member5`).
  - *Refer to `TODO.md` for exact task instructions and success criteria.*

---

## 9. Academic Integrity & Reproducibility
- All random splits and model initializations use `RANDOM_SEED = 42`.
- No mock predictions are fabricated; missing models are explicitly reported to the user.
- Real confusion matrices and classification reports are written to `results/`.

---

## 10. Project Deliverables & Reports

- **Milestone Project Report (PDF):** [`docs/IndicSent-M_Project_Report.pdf`](docs/IndicSent-M_Project_Report.pdf)
- **Literature Survey & Architecture Presentation (PPTX):** [`docs/IndicSent-M_Presentation.pptx`](docs/IndicSent-M_Presentation.pptx)
