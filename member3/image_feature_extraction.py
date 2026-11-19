"""
member3/image_feature_extraction.py
Extracts deep visual features from meme images using a pretrained ResNet backbone.
Operates with frozen weights in evaluation mode for fast, resource-friendly extraction.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGE_CONFIG
from member3.image_preprocessing import load_and_preprocess_image


class ImageFeatureExtractor:
    """
    Task 3.1: Configurable Deep Visual Feature Extractor.
    Supports comparison between:
    - ResNet-18 (512 dimensions)
    - ResNet-50 (2048 dimensions)
    - ViT-B-16 Vision Transformer (768 dimensions)
    Includes graceful offline fallback and optional meme text region masking.
    """

    def __init__(
        self,
        backbone: str = None,
        device: str = None,
        apply_masking: bool = False
    ):
        self.backbone_name = (backbone or IMAGE_CONFIG.get("backbone", "resnet18")).lower()
        self.apply_masking = apply_masking
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.has_pretrained_weights = False

        self._init_backbone()

    def _init_backbone(self):
        """Initialize selected visual backbone and strip classification head."""
        if self.backbone_name == "resnet50":
            try:
                weights = tv_models.ResNet50_Weights.DEFAULT
                self.model = tv_models.resnet50(weights=weights)
                self.has_pretrained_weights = True
            except Exception:
                print("Note: Pretrained ResNet-50 weights not downloaded. Using initialized ResNet-50 architecture.")
                self.model = tv_models.resnet50(weights=None)
            self.feature_dim = self.model.fc.in_features  # 2048
            self.model.fc = nn.Identity()

        elif self.backbone_name == "vit_b_16":
            try:
                weights = tv_models.ViT_B_16_Weights.DEFAULT
                self.model = tv_models.vit_b_16(weights=weights)
                self.has_pretrained_weights = True
            except Exception:
                print("Note: Pretrained ViT-B-16 weights not downloaded. Using initialized ViT-B-16 architecture.")
                self.model = tv_models.vit_b_16(weights=None)
            self.feature_dim = self.model.heads.head.in_features  # 768
            self.model.heads.head = nn.Identity()

        else:  # Default: resnet18
            self.backbone_name = "resnet18"
            try:
                weights = tv_models.ResNet18_Weights.DEFAULT
                self.model = tv_models.resnet18(weights=weights)
                self.has_pretrained_weights = True
            except Exception:
                print("Note: Pretrained ResNet-18 weights not downloaded. Using initialized ResNet-18 architecture.")
                self.model = tv_models.resnet18(weights=None)
            self.feature_dim = self.model.fc.in_features  # 512
            self.model.fc = nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features_single(
        self,
        image_input: Union[str, Path, Image.Image],
        apply_masking: Optional[bool] = None
    ) -> Optional[np.ndarray]:
        """Extract a 1D feature vector for a single image."""
        mask = self.apply_masking if apply_masking is None else apply_masking
        tensor = load_and_preprocess_image(image_input, apply_masking=mask)
        if tensor is None:
            return None

        tensor = tensor.to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat_np = feat.cpu().squeeze(0).numpy()
        return feat_np

    def extract_features_batch(
        self,
        image_paths: List[Union[str, Path]],
        apply_masking: Optional[bool] = None
    ) -> np.ndarray:
        """Extract features for a list of image paths. Imputes mean feature vector for invalid images."""
        features = []
        valid_indices = []

        for idx, path in enumerate(image_paths):
            f = self.extract_features_single(path, apply_masking=apply_masking)
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
    parser = argparse.ArgumentParser(description="Image Feature Extraction (Member 3)")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50", "vit_b_16"], default="resnet18", help="Vision backbone architecture")
    parser.add_argument("--mask-text", action="store_true", help="Apply meme visual region masking")
    args = parser.parse_args()

    extractor = ImageFeatureExtractor(backbone=args.backbone, apply_masking=args.mask_text)
    dummy_img = Image.new("RGB", (224, 224), color=(120, 200, 150))
    feat = extractor.extract_features_single(dummy_img)
    print(f"Backbone: {args.backbone.upper()} | Feature Dimension: {feat.shape} | Active: {extractor.has_pretrained_weights} | Masking: {args.mask_text}")
