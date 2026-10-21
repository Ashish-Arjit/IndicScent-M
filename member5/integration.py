"""
member5/integration.py
Unified inference manager for IndicSent-M. Coordinates Member 2 (text),
Member 3 (image), Member 4 (multimodal & conflict detection) into an integrated API.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR
from member2.text_model import TextSentimentClassifier
from member3.image_model import ImageSentimentClassifier
from member4.multimodal_model import MultimodalSentimentClassifier
from member4.conflict_detection import detect_modality_conflict


class IndicSentPipeline:
    """End-to-end inference engine coordinating text, image, and multimodal predictions."""

    def __init__(self):
        self.text_model = None
        self.image_model = None
        self.multimodal_model = None
        self._load_available_models()

    def _load_available_models(self):
        """Load any models that have been trained and saved on disk."""
        # 1. Text model
        text_path = MODELS_DIR / "text_model.pkl"
        if text_path.exists():
            try:
                self.text_model = TextSentimentClassifier().load()
            except Exception as e:
                print(f"Warning: Failed to load text model: {e}")

        # 2. Image model
        image_path = MODELS_DIR / "image_model.pkl"
        if image_path.exists():
            try:
                self.image_model = ImageSentimentClassifier().load()
            except Exception as e:
                print(f"Warning: Failed to load image model: {e}")

        # 3. Multimodal model
        mm_path = MODELS_DIR / "multimodal_model.pkl"
        if mm_path.exists():
            try:
                self.multimodal_model = MultimodalSentimentClassifier().load()
            except Exception as e:
                print(f"Warning: Failed to load multimodal model: {e}")

    def get_system_status(self) -> Dict[str, bool]:
        """Check availability of trained model checkpoints."""
        return {
            "text_model_loaded": self.text_model is not None,
            "image_model_loaded": self.image_model is not None,
            "multimodal_model_loaded": self.multimodal_model is not None
        }

    def analyze(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Run inference across all available modalities for the provided inputs.
        Detects conflicts if both text and image are supplied.
        """
        has_text = bool(text and text.strip())
        has_image = image is not None

        if not has_text and not has_image:
            return {
                "status": "error",
                "message": "Please provide at least text or an image for sentiment analysis."
            }

        response = {
            "status": "success",
            "input_modalities": {
                "has_text": has_text,
                "has_image": has_image
            },
            "text_prediction": None,
            "image_prediction": None,
            "multimodal_prediction": None,
            "conflict_diagnosis": None
        }

        # 1. Text-Only Inference
        if has_text:
            if self.text_model:
                preds = self.text_model.predict(text)
                response["text_prediction"] = preds[0]
            else:
                response["text_prediction"] = {"status": "Model not trained. Run python main.py --train-text"}

        # 2. Image-Only Inference
        if has_image:
            if self.image_model:
                preds = self.image_model.predict(image)
                response["image_prediction"] = preds[0]
            else:
                response["image_prediction"] = {"status": "Model not trained. Run python main.py --train-image"}

        # 3. Multimodal Inference & Conflict Detection (when both modalities are present)
        if has_text and has_image:
            if self.multimodal_model:
                mm_preds = self.multimodal_model.predict([text], [image])
                response["multimodal_prediction"] = mm_preds[0]
            else:
                response["multimodal_prediction"] = {"status": "Model not trained. Run python main.py --train-multimodal"}

            # Perform conflict diagnosis if both text & image predictions succeeded
            t_res = response.get("text_prediction")
            i_res = response.get("image_prediction")
            m_res = response.get("multimodal_prediction")

            if (t_res and "sentiment" in t_res) and (i_res and "sentiment" in i_res):
                m_valid = m_res if (m_res and "sentiment" in m_res) else None
                response["conflict_diagnosis"] = detect_modality_conflict(t_res, i_res, m_valid)

        return response


if __name__ == "__main__":
    pipeline = IndicSentPipeline()
    print("Pipeline Status:", pipeline.get_system_status())
