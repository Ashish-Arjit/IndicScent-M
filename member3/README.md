# Member 3: Image Sentiment Analysis

**Owner:** Surya Pratap Singh  
**Milestone:** Visual Feature Extraction, Pretrained ResNet Backbone, and Image-Only Sentiment Classification

---

## Responsibilities

1. **Image Preprocessing:** Ensure correct aspect resizing (224x224), RGB conversion, and torchvision ImageNet tensor normalization (`image_preprocessing.py`).
2. **Feature Extraction:** Utilize a pretrained ResNet-18 model to extract 512-dimensional penultimate layer feature representations with frozen weights (`image_feature_extraction.py`).
3. **Model Development:** Train and calibrate a linear classification head on top of visual feature embeddings (`image_model.py`).
4. **Evaluation:** Evaluate image-only classification performance against test memes, logging Accuracy, Macro-F1, Precision, Recall, and Confusion Matrices (`evaluate_image.py`).

---

## Exposed Functions & Classes

- `load_and_preprocess_image(image_input) -> torch.Tensor`: Loads and normalizes an image.
- `ImageFeatureExtractor`: Extracts 512-dim visual embeddings from single images or batches.
- `ImageSentimentClassifier`: Train/predict/save/load interface for image sentiment.
- `train_image_pipeline()`: Automates image model training on `data/processed/train.csv`.
- `evaluate_image_model()`: Computes and exports evaluation metrics to `results/`.

---

## Usage

```bash
# Train image sentiment classifier
python member3/image_model.py

# Evaluate on test set
python member3/evaluate_image.py
```
