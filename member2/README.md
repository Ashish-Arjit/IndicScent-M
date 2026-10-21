# Member 2: Text Sentiment Analysis

**Owner:** B.T. Ashish  
**Milestone:** Multilingual / Code-Mixed Text Preprocessing, Feature Extraction, and Text Classification

---

## Responsibilities

1. **Text Preprocessing:** Clean social media text, strip noise, handle code-mixing (Hinglish/Indian languages), and preserve sentiment-bearing emojis and markers (`text_preprocessing.py`).
2. **Feature Extraction:** Build an n-gram TF-IDF vectorizer suited for code-mixed Indian text (`text_feature_extraction.py`).
3. **Model Development:** Train a calibrated 3-class sentiment classifier (`text_model.py`) outputting probabilities for `negative`, `neutral`, and `positive`.
4. **Evaluation:** Evaluate performance on the test split, producing Accuracy, Macro-F1, Precision, Recall, Confusion Matrix, and detailed predictions (`evaluate_text.py`).

---

## Exposed Functions & Classes

- `clean_text(text: str) -> str`: Normalizes single text string.
- `preprocess_corpus(texts) -> List[str]`: Cleans a list/Series of texts.
- `TextFeatureExtractor`: Fits and transforms text into feature matrices.
- `TextSentimentClassifier`: Full train/predict/save/load interface for text sentiment.
- `train_text_pipeline()`: Automates text training on `data/processed/train.csv`.
- `evaluate_text_model()`: Computes and exports evaluation metrics to `results/`.

---

## Usage

```bash
# Train text sentiment classifier
python member2/text_model.py

# Evaluate on test set
python member2/evaluate_text.py
```
