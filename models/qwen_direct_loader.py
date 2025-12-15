"""
Qwen3-VL Direct Model Loader (No vLLM Server Required)

Optimized Hugging Face implementation with batching and memory efficiency.
Primary inference method for the project.
"""
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from typing import Optional, List, Dict, Any, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class Qwen3VLDirectDetector:
    """
    Optimized Qwen3-VL detector using Hugging Face Transformers.

    Features:
    - True batch processing for better GPU utilization
    - Optional 8-bit quantization for lower memory usage
    - Memory-efficient inference with gradient disabling
    - Comprehensive error handling and logging
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Thinking",  # Same as official SAM3 agent
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False
    ):
        """
        Initialize Qwen3-VL detector with direct model loading.

        Args:
            model_name: Hugging Face model ID
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            use_quantization: Use 8-bit quantization to reduce memory usage by ~50%
            low_memory: Enable additional memory optimizations
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization
        self.low_memory = low_memory

        print(f"Loading {model_name} directly (Hugging Face Transformers)...")
        print(f"Device: {self.device}")
        print(f"Quantization: {'Enabled (8-bit)' if use_quantization else 'Disabled (FP16/FP32)'}")
        print(f"Low memory mode: {'Enabled' if low_memory else 'Disabled'}")

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
            logger.warning(f"AutoProcessor failed ({e}), using AutoImageProcessor...")
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

        # Load model with optional quantization
        self.model = self._load_model()
        self.model.eval()

        print("✓ Model loaded successfully!")
        if self.device == "cuda":
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def _load_model(self):
        """Load model with appropriate optimizations."""
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": True,
        }

        # Apply quantization if requested
        if self.use_quantization and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
                print("Using 8-bit quantization (reduces memory by ~50%)...")
            except ImportError:
                logger.warning("bitsandbytes not installed. Install with: pip install bitsandbytes")
                logger.warning("Falling back to standard precision...")
                self.use_quantization = False

        # Standard loading without quantization
        if not self.use_quantization:
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None

        # Low memory optimizations
        if self.low_memory:
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForVision2Seq.from_pretrained(**load_kwargs)

        if self.device == "cpu" and not self.use_quantization:
            model = model.to(self.device)

        return model

    def detect(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Run detection on a single image with a custom prompt.

        Args:
            image: PIL Image or path to image
            prompt: Detection prompt
            max_new_tokens: Max tokens to generate

        Returns:
            dict: {
                "text": response_text,
                "raw_output": raw_response,
                "success": bool
            }
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')

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
            )

            # Move inputs to same device as model
            if self.use_quantization:
                # With quantization, model is on cuda via device_map="auto"
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Generate with gradient disabled for memory efficiency
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
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

            # Clear GPU cache if in low memory mode
            if self.low_memory and self.device == "cuda":
                torch.cuda.empty_cache()

            return {
                "text": response,
                "raw_output": response,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return {
                "text": "",
                "raw_output": "",
                "success": False,
                "error": str(e)
            }

    def batch_detect(
        self,
        images: List[Union[Image.Image, str]],
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1024,
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Run detection on multiple images with true batch processing.

        This method processes images in batches for better GPU utilization
        compared to sequential processing.

        Args:
            images: List of PIL Images or paths to images
            prompts: Single prompt (used for all) or list of prompts (one per image)
            max_new_tokens: Max tokens to generate
            batch_size: Number of images to process at once (reduce if OOM)

        Returns:
            list: List of detection results, one per image

        Example:
            >>> images = [Image.open(f"frame_{i}.jpg") for i in range(10)]
            >>> prompt = "Detect infrastructure defects"
            >>> results = detector.batch_detect(images, prompt, batch_size=4)
        """
        results = []

        # Load images if paths provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                try:
                    loaded_images.append(Image.open(img).convert('RGB'))
                except Exception as e:
                    logger.error(f"Failed to load image {img}: {e}")
                    loaded_images.append(None)
            else:
                loaded_images.append(img)

        # Handle single prompt for all images
        if isinstance(prompts, str):
            prompts = [prompts] * len(loaded_images)

        # Process in batches
        for i in range(0, len(loaded_images), batch_size):
            batch_images = loaded_images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            # Filter out None images
            valid_pairs = [(img, prompt) for img, prompt in zip(batch_images, batch_prompts) if img is not None]

            if not valid_pairs:
                # All images in batch failed to load
                results.extend([{"text": "", "raw_output": "", "success": False, "error": "Failed to load image"}] * len(batch_images))
                continue

            valid_images, valid_prompts = zip(*valid_pairs)

            try:
                # Prepare batch messages
                messages_batch = [
                    [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                    for img, prompt in zip(valid_images, valid_prompts)
                ]

                # Apply chat template to all
                texts = [
                    self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    for msgs in messages_batch
                ]

                # Process batch
                inputs = self.processor(
                    text=texts,
                    images=list(valid_images),
                    padding=True,
                    return_tensors="pt"
                )

                # Move inputs to same device as model
                if self.use_quantization:
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                # Decode all outputs
                for j, (in_ids, out_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
                    trimmed = out_ids[len(in_ids):]
                    response = self.tokenizer.decode(
                        trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    results.append({
                        "text": response,
                        "raw_output": response,
                        "success": True
                    })

                # Clear GPU cache after batch
                if self.low_memory and self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                # Add error results for this batch
                results.extend([
                    {"text": "", "raw_output": "", "success": False, "error": str(e)}
                    for _ in range(len(valid_images))
                ])

        return results

    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self.device == "cuda":
            del self.model
            torch.cuda.empty_cache()
            print("GPU memory cleared")


def get_direct_detector_config(
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    use_quantization: bool = False,
    low_memory: bool = False
) -> Dict[str, Any]:
    """
    Get configuration for direct detector.

    Args:
        model_name: Hugging Face model ID
        use_quantization: Enable 8-bit quantization
        low_memory: Enable low memory optimizations

    Returns:
        dict: Configuration dictionary
    """
    return {
        "mode": "direct",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float16" if torch.cuda.is_available() else "float32",
        "use_quantization": use_quantization,
        "low_memory": low_memory
    }
