"""
Model loaders for SAM3 and Qwen3-VL
"""
from .qwen_direct_loader import Qwen3VLDirectDetector

# SAM3 is optional - only import if sam3 package is installed
try:
    from .sam3_loader import load_sam3_model, get_sam3_config
    SAM3_AVAILABLE = True
except ImportError:
    load_sam3_model = None
    get_sam3_config = None
    SAM3_AVAILABLE = False

__all__ = [
    "load_sam3_model",
    "get_sam3_config",
    "Qwen3VLDirectDetector",
    "SAM3_AVAILABLE"
]
