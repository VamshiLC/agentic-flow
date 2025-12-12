"""
Qwen3-VL Direct Model Loader (No vLLM Server Required)

Alternative to vLLM server - loads Qwen3-VL directly using transformers.
Simpler setup but slower inference.
"""
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from typing import Optional


class Qwen3VLDirectDetector:
    """
    Direct Qwen3-VL detector without vLLM server.

    Based on user's detector.py implementation.
    Simpler setup, no server required, but slower than vLLM.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = None
    ):
        """
        Initialize Qwen3-VL detector with direct model loading.

        Args:
            model_name: Hugging Face model ID
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name} directly (no vLLM server)...")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: AutoProcessor failed ({e}), using AutoImageProcessor...")
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("✓ Model loaded successfully!")

    def detect(self, image, prompt: str, max_new_tokens: int = 1024):
        """
        Run detection on an image with a custom prompt.

        Args:
            image: PIL Image
            prompt: Detection prompt
            max_new_tokens: Max tokens to generate

        Returns:
            dict: {"text": response_text, "raw_output": raw_response}
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return {
            "text": response,
            "raw_output": response
        }

    def batch_detect(self, images, prompts, max_new_tokens: int = 1024):
        """
        Run detection on multiple images with multiple prompts.

        Args:
            images: List of PIL Images
            prompts: List of prompts (one per image or one for all)
            max_new_tokens: Max tokens to generate

        Returns:
            list: List of detection results
        """
        results = []

        # Handle single prompt for all images
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        for image, prompt in zip(images, prompts):
            result = self.detect(image, prompt, max_new_tokens)
            results.append(result)

        return results


def get_direct_detector_config(model_name="Qwen/Qwen3-VL-4B-Instruct"):
    """
    Get configuration for direct detector.

    Returns:
        dict: Configuration dictionary
    """
    return {
        "mode": "direct",  # Not vLLM
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float16" if torch.cuda.is_available() else "float32"
    }
