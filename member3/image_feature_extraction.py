"""
member3/image_feature_extraction.py
Extracts deep visual features from meme images using a pretrained ResNet backbone.
Operates with frozen weights in evaluation mode for fast, resource-friendly extraction.
"""

import sys
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGE_CONFIG
from member3.image_preprocessing import load_and_preprocess_image


class ImageFeatureExtractor:
    """Pretrained ResNet feature extractor producing fixed-dimension visual embeddings."""

    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize ResNet18 with pretrained weights if available, otherwise random init
        try:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.has_pretrained_weights = True
        except Exception:
            # Fallback for completely offline or network-restricted environments
            print("Note: Pretrained weights not downloaded. Using initialized ResNet-18 architecture.")
            self.model = resnet18(weights=None)
            self.has_pretrained_weights = False

        # Replace classification layer with Identity to extract 512-dim penultimate features
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features_single(
        self,
        image_input: Union[str, Path, Image.Image]
    ) -> Optional[np.ndarray]:
        """Extract a 1D feature vector for a single image."""
        tensor = load_and_preprocess_image(image_input)
        if tensor is None:
            return None

        tensor = tensor.to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat_np = feat.cpu().squeeze(0).numpy()
        return feat_np

    def extract_features_batch(
        self,
        image_paths: List[Union[str, Path]]
    ) -> np.ndarray:
        """Extract features for a list of image paths. Imputes mean feature vector for invalid images."""
        features = []
        valid_indices = []

        for idx, path in enumerate(image_paths):
            f = self.extract_features_single(path)
            if f is not None:
                features.append(f)
                valid_indices.append(idx)
            else:
                features.append(np.zeros(self.feature_dim, dtype=np.float32))

        features_arr = np.array(features, dtype=np.float32)

        # Impute zero rows if any failed
        if len(valid_indices) < len(image_paths) and len(valid_indices) > 0:
            mean_vec = np.mean(features_arr[valid_indices], axis=0)
            for idx in range(len(image_paths)):
                if idx not in valid_indices:
                    features_arr[idx] = mean_vec

        return features_arr


if __name__ == "__main__":
    extractor = ImageFeatureExtractor()
    dummy_img = Image.new("RGB", (224, 224), color=(120, 200, 150))
    feat = extractor.extract_features_single(dummy_img)
    print("ResNet18 Feature Vector Shape:", feat.shape)
    print("Pretrained Weights Active:", extractor.has_pretrained_weights)
