#!/usr/bin/env python3
"""
SAM3-Only Single Frame Inference
Processes single images using SAM3 text prompts for detection and segmentation.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import from existing modules
from prompts.category_prompts import CATEGORY_PROMPTS


class SAM3OnlyFrameProcessor:
    """
    Processes single frames using SAM3 text prompts only.

    Unlike the agentic flow (Qwen + SAM3), this uses SAM3 directly
    with text prompts for both detection and segmentation.
    """

    def __init__(
        self,
        model,
        processor,
        categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        device: str = "cuda",
    ):
        """
        Initialize SAM3-only frame processor.

        Args:
            model: SAM3 Video model
            processor: SAM3 Video processor
            categories: List of category names to detect (None = all)
            confidence_threshold: Minimum confidence score
            device: Device for processing
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Get categories to detect
        if categories is None:
            # Use all available categories
            self.categories = list(CATEGORY_PROMPTS.keys())
        else:
            self.categories = categories

        # Build text prompts for SAM3
        self.text_prompts = self._build_text_prompts()

        print(f"SAM3OnlyFrameProcessor initialized:")
        print(f"  Categories: {len(self.categories)}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Device: {device}")

    def _build_text_prompts(self) -> List[str]:
        """
        Build text prompts for SAM3 detection.

        Uses simplified prompts optimized for SAM3's text understanding.
        """
        prompts = []
        for category in self.categories:
            if category in CATEGORY_PROMPTS:
                # Get the main category name - SAM3 works better with simple prompts
                category_info = CATEGORY_PROMPTS[category]
                prompt = category_info.get("short_name", category)
                prompts.append(prompt)
            else:
                prompts.append(category)

        return prompts

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image with SAM3 text prompts.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with detections and metadata
        """
        # Load image
        image = self._load_image(image_path)

        # Run detection with all prompts
        detections = self._detect_with_prompts(image)

        # Format results
        result = {
            "image_path": str(image_path),
            "image_size": image.size,
            "num_detections": len(detections),
            "detections": detections,
            "categories_searched": self.categories,
            "prompts_used": self.text_prompts,
        }

        return result

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image."""
        image = Image.open(image_path).convert("RGB")
        return image

    def _detect_with_prompts(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run SAM3 detection with text prompts.

        Args:
            image: PIL Image

        Returns:
            List of detections
        """
        all_detections = []

        # Convert PIL image to frames format (single frame as video)
        frames = [image]

        # Process each prompt separately
        for prompt_idx, (category, prompt) in enumerate(zip(self.categories, self.text_prompts)):
            try:
                # Initialize video inference session (even for single image)
                inference_session = self.processor.init_video_session(
                    video=frames,
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.float32,
                )

                # Add text prompt
                inference_session = self.processor.add_text_prompt(
                    inference_session=inference_session,
                    text=prompt,
                )

                # Process single frame
                with torch.no_grad():
                    for model_outputs in self.model.propagate_in_video_iterator(
                        inference_session=inference_session,
                        max_frame_num_to_track=1  # Single frame
                    ):
                        processed = self.processor.postprocess_outputs(
                            inference_session, model_outputs
                        )

                        # Extract detections
                        if processed.get('object_ids') is not None and len(processed['object_ids']) > 0:
                            for i, obj_id in enumerate(processed['object_ids']):
                                # Get confidence score
                                score = float(processed['scores'][i]) if 'scores' in processed else 0.5

                                # Skip low confidence detections
                                if score < self.confidence_threshold:
                                    continue

                                # Get bounding box (XYXY format)
                                bbox = processed['boxes'][i].tolist() if 'boxes' in processed else [0, 0, 0, 0]

                                # Get mask
                                mask = None
                                if 'masks' in processed and processed['masks'] is not None:
                                    mask_tensor = processed['masks'][i]
                                    mask = mask_tensor.cpu().numpy()
                                    if len(mask.shape) > 2:
                                        mask = mask.squeeze()

                                # Create detection dictionary
                                detection = {
                                    "label": category,
                                    "category": category,
                                    "prompt_used": prompt,
                                    "bbox": bbox,
                                    "confidence": score,
                                    "object_id": int(obj_id),
                                    "mask": mask,
                                    "has_mask": mask is not None,
                                }

                                all_detections.append(detection)

                # Clear GPU memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Warning: Error processing prompt '{prompt}': {e}")
                continue

        print(f"  Found {len(all_detections)} detections")
        return all_detections

    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of results (one per image)
        """
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            result = self.process_image(image_path)
            results.append(result)

        return results


def create_sam3_only_processor(
    model,
    processor,
    categories: Optional[List[str]] = None,
    confidence_threshold: float = 0.3,
    device: str = "cuda",
) -> SAM3OnlyFrameProcessor:
    """
    Convenience function to create SAM3-only processor.

    Args:
        model: SAM3 Video model
        processor: SAM3 Video processor
        categories: Categories to detect (None = all)
        confidence_threshold: Minimum confidence
        device: Processing device

    Returns:
        SAM3OnlyFrameProcessor instance
    """
    return SAM3OnlyFrameProcessor(
        model=model,
        processor=processor,
        categories=categories,
        confidence_threshold=confidence_threshold,
        device=device,
    )
