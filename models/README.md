# IndicSent-M Trained Models Directory

This directory stores serialized model weights, vectorizers, and scalers produced during training.

## Expected Artifacts

- `text_vectorizer.pkl`: Fitted scikit-learn TF-IDF vectorizer (Member 2).
- `text_model.pkl`: Trained text-only sentiment classifier (Member 2).
- `image_model.pkl`: Trained image-only sentiment classifier (Member 3).
- `multimodal_model.pkl`: Trained multimodal fusion classifier (Member 4).
- `fusion_scaler.pkl`: Feature normalization scaler for concatenated representations.

## Note for Students
If using PyTorch or fine-tuned Transformer models (such as IndicBERT or ResNet fine-tuning), store model state dicts as `.pt` or `.bin` in this folder.
