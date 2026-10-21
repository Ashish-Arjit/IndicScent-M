"""
conftest.py - PyTest Root Configuration for IndicSent-M
Ensures PyTorch and Windows OpenMP libraries are initialized before test collection.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass
