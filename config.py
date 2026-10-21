import os
import sys
from pathlib import Path

# Fix Windows OpenMP and DLL collisions between PyTorch and Scikit-Learn/NumPy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
except Exception:
    pass


# Base Paths (dynamically resolved relative to this config file)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_MEMES_DIR = DATA_DIR / "sample_memes"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Standard 3-class Sentiment Labels
LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["negative", "neutral", "positive"]
NUM_CLASSES = 3

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Dataset Split Ratios
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# Member 2: Text Preprocessing & Model Hyperparameters
TEXT_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "analyzer": "word",
    "sublinear_tf": True,
    "c_param": 1.0,
    "max_iter": 1000,
    "transformer_model_name": "ai4bharat/indic-bert",
    "max_seq_length": 128
}

# Member 3: Image Preprocessing & Feature Extraction Hyperparameters
IMAGE_CONFIG = {
    "image_size": (224, 224),
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "backbone": "resnet18",
    "feature_dim": 512,  # Penultimate layer representation for ResNet18
    "c_param": 1.0,
    "max_iter": 1000
}

# Member 4: Multimodal Fusion & Conflict Detection Hyperparameters
FUSION_CONFIG = {
    "fusion_method": "concat",  # 'concat', 'weighted', or 'late_fusion'
    "c_param": 1.0,
    "max_iter": 1000,
    "conflict_confidence_threshold": 0.45  # Minimum confidence to flag high-severity conflict
}

# Member 5: Application Settings
APP_CONFIG = {
    "title": "IndicSent-M: Multimodal Sentiment Analysis",
    "theme_color": "#1E3A8A",
    "host": "0.0.0.0",
    "port": 8501
}


def ensure_directories():
    """Ensure that all necessary project directories exist."""
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_MEMES_DIR, MODELS_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"IndicSent-M directories initialized at: {BASE_DIR}")
    print(f"Classes: {LABEL_MAP}")
