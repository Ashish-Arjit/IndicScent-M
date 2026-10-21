"""
member3/image_preprocessing.py
Handles image loading, RGB conversion, resizing, and torchvision tensor normalization.
Ensures identical preprocessing during training and real-time inference.
"""

import sys
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import torch
from torchvision import transforms

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGE_CONFIG


def get_image_transforms() -> transforms.Compose:
    """Standard ResNet preprocessing transform pipeline."""
    return transforms.Compose([
        transforms.Resize(IMAGE_CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGE_CONFIG["mean"],
            std=IMAGE_CONFIG["std"]
        )
    ])


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image]
) -> Optional[torch.Tensor]:
    """
    Load an image from file path or PIL instance, convert to RGB,
    and apply standard normalization transforms.
    Returns a 4D tensor (1, 3, H, W) or None if corrupt.
    """
    transform = get_image_transforms()
    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                print(f"Warning: Image file not found: {path}")
                return None
            img = Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            print(f"Unsupported image input type: {type(image_input)}")
            return None

        # Apply transforms and add batch dimension
        tensor = transform(img).unsqueeze(0)
        return tensor

    except Exception as e:
        print(f"Error preprocessing image {image_input}: {e}")
        return None


if __name__ == "__main__":
    # Test on a dummy blank image
    dummy_img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    tensor = load_and_preprocess_image(dummy_img)
    print("Preprocessed Tensor Shape:", tensor.shape)
    print("Tensor Mean:", tensor.mean().item(), "Std:", tensor.std().item())
