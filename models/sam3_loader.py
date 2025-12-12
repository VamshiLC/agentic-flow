"""
SAM3 Model Loader

Loads the SAM3 model following Meta's official pattern:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb
"""
import os
import torch
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_sam3_model(confidence_threshold=0.5, device="cuda"):
    """
    Load SAM3 model with processor and optimizations.

    Args:
        confidence_threshold: Minimum confidence score for mask predictions (0.0-1.0)
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Sam3Processor: Configured SAM3 processor ready for inference

    Notes:
        - Enables TF32 for Ampere GPUs (faster matrix multiplication)
        - Uses bfloat16 precision for reduced memory usage
        - Runs in inference mode (disables gradients)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Loading SAM3 model on {device}...")

    # Enable TF32 for Ampere GPUs (3000 series and newer)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable bfloat16 autocast for reduced memory usage
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Enable inference mode (disables gradient computation)
    torch.inference_mode().__enter__()

    # Build SAM3 model
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    if not os.path.exists(bpe_path):
        raise FileNotFoundError(
            f"SAM3 BPE vocabulary file not found at {bpe_path}. "
            "Make sure SAM3 is installed correctly."
        )

    model = build_sam3_image_model(bpe_path=bpe_path)

    # Create processor with confidence threshold
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    print(f"SAM3 model loaded successfully (confidence threshold: {confidence_threshold})")

    return processor


def get_sam3_config():
    """
    Get default SAM3 configuration.

    Returns:
        dict: Configuration dictionary for SAM3
    """
    return {
        "confidence_threshold": 0.5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16" if torch.cuda.is_available() else "float32"
    }
