#!/usr/bin/env python3
"""
SAM3-Only Video Processor
Processes videos using SAM3 text prompts for detection and segmentation.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers.video_utils import load_video


class SAM3OnlyVideoProcessor:
    """
    Processes videos using SAM3 text prompts only.

    Handles frame extraction, batch processing, and video-level results aggregation.
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
        Initialize SAM3-only video processor.

        Args:
            model: SAM3 Video model
            processor: SAM3 Video processor
            categories: List of category names to detect
            confidence_threshold: Minimum confidence score
            device: Device for processing
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Import categories from existing prompts
        from prompts.category_prompts import CATEGORY_PROMPTS

        if categories is None:
            self.categories = list(CATEGORY_PROMPTS.keys())
        else:
            self.categories = categories

        self.category_prompts = CATEGORY_PROMPTS
        self.text_prompts = self._build_text_prompts()

        print(f"SAM3OnlyVideoProcessor initialized:")
        print(f"  Categories: {len(self.categories)}")
        print(f"  Confidence threshold: {confidence_threshold}")

    def _build_text_prompts(self) -> List[str]:
        """Build optimized text prompts for SAM3."""
        prompts = []
        for category in self.categories:
            if category in self.category_prompts:
                prompt = self.category_prompts[category].get("short_name", category)
                prompts.append(prompt)
            else:
                prompts.append(category)
        return prompts

    def process_video(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file with SAM3 text prompts.

        Args:
            video_path: Path to video file
            target_fps: Target FPS for frame extraction (None = use original)
            max_frames: Maximum frames to process (None = all)

        Returns:
            Dictionary with all detections and metadata
        """
        print(f"\nProcessing video: {video_path}")

        # Load video frames
        frames, fps, resolution = self._load_video_frames(video_path, target_fps)

        # Limit frames if specified
        if max_frames:
            frames = frames[:max_frames]
            print(f"  Limited to {len(frames)} frames")

        # Process frames with SAM3
        all_detections = self._process_frames_with_sam3(frames, fps)

        # Aggregate results
        result = {
            "video_path": str(video_path),
            "video_name": Path(video_path).stem,
            "fps": fps,
            "total_frames": len(frames),
            "resolution": resolution,
            "num_detections": len(all_detections),
            "detections": all_detections,
            "categories_searched": self.categories,
        }

        return result

    def _load_video_frames(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
    ) -> tuple:
        """
        Load video frames using transformers or OpenCV fallback.

        Args:
            video_path: Path to video
            target_fps: Target FPS (None = original)

        Returns:
            Tuple of (frames, fps, resolution)
        """
        video_path = Path(video_path)

        # Try transformers video loader first
        try:
            frames, video_info = load_video(str(video_path))

            # Get metadata using OpenCV
            cap = cv2.VideoCapture(str(video_path))
            original_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Subsample frames if target_fps specified
            if target_fps and target_fps < original_fps:
                frame_interval = int(original_fps / target_fps)
                frames = frames[::frame_interval]
                fps = target_fps
            else:
                fps = original_fps

            print(f"  Loaded {len(frames)} frames at {fps:.1f} FPS")
            print(f"  Resolution: {width}x{height}")

            return frames, fps, (width, height)

        except Exception as e:
            print(f"  Transformers loader failed, using OpenCV: {e}")
            return self._load_video_opencv(video_path, target_fps)

    def _load_video_opencv(
        self,
        video_path: Path,
        target_fps: Optional[float] = None,
    ) -> tuple:
        """Fallback: Load video using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame interval
        if target_fps and target_fps < original_fps:
            frame_interval = int(original_fps / target_fps)
            fps = target_fps
        else:
            frame_interval = 1
            fps = original_fps

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

            frame_count += 1

        cap.release()

        print(f"  Loaded {len(frames)} frames at {fps:.1f} FPS")
        print(f"  Resolution: {width}x{height}")

        return frames, fps, (width, height)

    def _process_frames_with_sam3(
        self,
        frames: List[Image.Image],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Process video frames with SAM3 text prompts.

        Args:
            frames: List of PIL Images
            fps: Frame rate

        Returns:
            List of all detections across all frames
        """
        all_detections = []

        print(f"\nRunning SAM3 detection with {len(self.text_prompts)} prompts...")

        # Process each prompt separately
        for prompt_idx, (category, prompt) in enumerate(zip(self.categories, self.text_prompts)):
            print(f"  [{prompt_idx + 1}/{len(self.text_prompts)}] Processing: '{prompt}'")

            try:
                # Initialize video inference session
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

                # Process all frames
                prompt_detections = 0
                with torch.no_grad():
                    for model_outputs in self.model.propagate_in_video_iterator(
                        inference_session=inference_session,
                        max_frame_num_to_track=len(frames)
                    ):
                        processed = self.processor.postprocess_outputs(
                            inference_session, model_outputs
                        )

                        frame_idx = model_outputs.frame_idx

                        # Extract detections from this frame
                        if processed.get('object_ids') is not None and len(processed['object_ids']) > 0:
                            for i, obj_id in enumerate(processed['object_ids']):
                                score = float(processed['scores'][i]) if 'scores' in processed else 0.5

                                # Skip low confidence
                                if score < self.confidence_threshold:
                                    continue

                                bbox = processed['boxes'][i].tolist() if 'boxes' in processed else [0, 0, 0, 0]

                                # Get mask
                                mask = None
                                if 'masks' in processed and processed['masks'] is not None:
                                    mask_tensor = processed['masks'][i]
                                    mask = mask_tensor.cpu().numpy()
                                    if len(mask.shape) > 2:
                                        mask = mask.squeeze()

                                detection = {
                                    "frame_number": frame_idx,
                                    "timestamp": frame_idx / fps,
                                    "label": category,
                                    "category": category,
                                    "prompt_used": prompt,
                                    "confidence": score,
                                    "bbox": bbox,
                                    "mask": mask,
                                    "has_mask": mask is not None,
                                    "object_id": int(obj_id),
                                    "tracking_id": f"{prompt_idx}_{obj_id}",
                                }

                                all_detections.append(detection)
                                prompt_detections += 1

                print(f"    → Found {prompt_detections} detections")

                # Clear GPU memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    → Error: {e}")
                continue

        print(f"\nTotal detections: {len(all_detections)}")
        return all_detections


def create_sam3_only_video_processor(
    model,
    processor,
    categories: Optional[List[str]] = None,
    confidence_threshold: float = 0.3,
    device: str = "cuda",
) -> SAM3OnlyVideoProcessor:
    """
    Convenience function to create SAM3-only video processor.

    Args:
        model: SAM3 Video model
        processor: SAM3 Video processor
        categories: Categories to detect
        confidence_threshold: Minimum confidence
        device: Processing device

    Returns:
        SAM3OnlyVideoProcessor instance
    """
    return SAM3OnlyVideoProcessor(
        model=model,
        processor=processor,
        categories=categories,
        confidence_threshold=confidence_threshold,
        device=device,
    )
