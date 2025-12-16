#!/usr/bin/env python3
"""
SAM3 Text Prompt Model Loader
Loads SAM3 Video model for text-prompted detection and segmentation.
"""

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from typing import Optional


class SAM3TextPromptLoader:
    """
    Loader for SAM3 Video model with text prompting capabilities.

    Unlike the box-prompted SAM3 used in the agentic flow, this version
    uses text prompts directly for detection and segmentation.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize SAM3 text-prompt model loader.

        Args:
            model_name: Hugging Face model ID
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            dtype: Model precision (float32 recommended for accuracy)
        """
        self.model_name = model_name
        self.dtype = dtype

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.processor = None

        print(f"SAM3TextPromptLoader initialized:")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def load(self):
        """Load the SAM3 model and processor."""
        if self.model is not None:
            print("Model already loaded, skipping...")
            return self.model, self.processor

        print(f"\nLoading SAM3 Video model from {self.model_name}...")
        try:
            # Load processor
            self.processor = Sam3VideoProcessor.from_pretrained(self.model_name)
            print("  ✓ Processor loaded")

            # Load model
            self.model = Sam3VideoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device, dtype=self.dtype)
            self.model.eval()
            print(f"  ✓ Model loaded to {self.device}")

            return self.model, self.processor

        except Exception as e:
            print(f"\n✗ Error loading SAM3 model: {e}")
            print("\nTroubleshooting:")
            print("1. Install latest transformers: pip install --upgrade transformers")
            print("2. Login to Hugging Face: huggingface-cli login")
            print("3. Accept model terms: https://huggingface.co/facebook/sam3")
            raise

    def get_model(self):
        """Get the loaded model (loads if not already loaded)."""
        if self.model is None:
            self.load()
        return self.model

    def get_processor(self):
        """Get the loaded processor (loads if not already loaded)."""
        if self.processor is None:
            self.load()
        return self.processor

    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'device': str(self.device)
            }
        return {'device': 'cpu'}


def load_sam3_text_prompt_model(
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Convenience function to load SAM3 text-prompt model.

    Args:
        device: Device to load model on
        dtype: Model precision

    Returns:
        Tuple of (model, processor, loader)
    """
    loader = SAM3TextPromptLoader(device=device, dtype=dtype)
    model, processor = loader.load()
    return model, processor, loader
