"""
SAM3 Video Tracker for License Plate Detection and Tracking

Uses SAM3's built-in video tracking with text prompts.
Provides consistent object IDs across video frames.

Falls back to IoU-based tracking if SAM3 video is unavailable.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Check SAM3 availability
SAM3_VIDEO_AVAILABLE = False
SAM3_IMAGE_AVAILABLE = False

try:
    from sam3 import build_sam3_video_model
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    SAM3_VIDEO_AVAILABLE = True
    logger.info("SAM3 Video tracking available")
except ImportError:
    logger.warning("SAM3 Video not available, will try image model")

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_IMAGE_AVAILABLE = True
    logger.info("SAM3 Image model available")
except ImportError:
    logger.warning("SAM3 Image model not available")

# Try Hugging Face transformers as alternative
HF_SAM3_AVAILABLE = False
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    HF_SAM3_AVAILABLE = True
    logger.info("Hugging Face SAM3 Video available")
except ImportError:
    pass


class SAM3PlateTracker:
    """
    SAM3-based license plate tracker.

    Uses SAM3's text-prompted detection and built-in video tracking
    to maintain consistent plate IDs across frames.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        text_prompt: str = "license plate"
    ):
        """
        Initialize SAM3 plate tracker.

        Args:
            device: Device to run on (cuda/cpu)
            confidence_threshold: Minimum confidence for detections
            text_prompt: Text prompt for plate detection
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.text_prompt = text_prompt

        self.model = None
        self.processor = None
        self.video_predictor = None
        self.mode = None  # 'video', 'image', or 'hf'

        # Track state for video processing
        self.is_initialized = False
        self.frame_index = 0
        self.tracked_objects = {}  # {track_id: {'masks': [], 'bboxes': [], 'frames': []}}

        self._load_model()

    def _load_model(self):
        """Load SAM3 model based on availability."""

        # Priority 1: SAM3 Video model (best for tracking)
        if SAM3_VIDEO_AVAILABLE:
            try:
                print("Loading SAM3 Video model for tracking...")
                self.model = build_sam3_video_model()
                self.model.to(self.device)
                self.model.eval()
                self.video_predictor = Sam3VideoPredictor(self.model)
                self.mode = 'video'
                print("✓ SAM3 Video model loaded (native tracking)")
                return
            except Exception as e:
                logger.warning(f"Failed to load SAM3 Video: {e}")

        # Priority 2: Hugging Face SAM3 Video
        if HF_SAM3_AVAILABLE:
            try:
                print("Loading Hugging Face SAM3 Video model...")
                self.model = Sam3VideoModel.from_pretrained("facebook/sam3-hiera-large")
                self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3-hiera-large")
                self.model.to(self.device)
                self.model.eval()
                self.mode = 'hf'
                print("✓ Hugging Face SAM3 Video model loaded")
                return
            except Exception as e:
                logger.warning(f"Failed to load HF SAM3 Video: {e}")

        # Priority 3: SAM3 Image model (per-frame, no native tracking)
        if SAM3_IMAGE_AVAILABLE:
            try:
                print("Loading SAM3 Image model (per-frame detection)...")
                import sam3
                import os
                sam3_root = os.path.dirname(sam3.__file__)
                bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

                self.model = build_sam3_image_model(bpe_path=bpe_path)
                self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
                self.mode = 'image'
                print("✓ SAM3 Image model loaded (will use IoU for tracking)")
                return
            except Exception as e:
                logger.warning(f"Failed to load SAM3 Image: {e}")

        # No SAM3 available
        self.mode = None
        print("⚠ No SAM3 model available - will use Qwen3-VL for detection")

    def is_available(self) -> bool:
        """Check if SAM3 is available."""
        return self.mode is not None

    def has_native_tracking(self) -> bool:
        """Check if native video tracking is available."""
        return self.mode in ['video', 'hf']

    def reset(self):
        """Reset tracker for new video."""
        self.is_initialized = False
        self.frame_index = 0
        self.tracked_objects = {}

        if self.video_predictor is not None:
            try:
                self.video_predictor.reset_state()
            except:
                pass

    def init_video(self, first_frame: Union[Image.Image, np.ndarray]):
        """
        Initialize video tracking with first frame.

        Args:
            first_frame: First frame of video
        """
        if isinstance(first_frame, np.ndarray):
            first_frame = Image.fromarray(first_frame)

        if self.mode == 'video' and self.video_predictor:
            try:
                # Initialize SAM3 video predictor with text prompt
                self.video_predictor.set_image(first_frame)
                self.video_predictor.set_text_prompt(self.text_prompt)
                self.is_initialized = True
                logger.info("SAM3 Video tracker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SAM3 video: {e}")
                self.is_initialized = False
        else:
            self.is_initialized = True

        self.frame_index = 0

    def detect_and_track(
        self,
        frame: Union[Image.Image, np.ndarray],
        frame_idx: Optional[int] = None
    ) -> List[Dict]:
        """
        Detect and track plates in a frame.

        Args:
            frame: Current video frame
            frame_idx: Frame index (auto-incremented if not provided)

        Returns:
            List of detections with track_id, bbox, mask
        """
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        if frame_idx is not None:
            self.frame_index = frame_idx

        detections = []

        if self.mode == 'video':
            detections = self._detect_video_mode(frame)
        elif self.mode == 'hf':
            detections = self._detect_hf_mode(frame)
        elif self.mode == 'image':
            detections = self._detect_image_mode(frame)

        self.frame_index += 1
        return detections

    def _detect_video_mode(self, frame: Image.Image) -> List[Dict]:
        """Detection using SAM3 native video tracking."""
        detections = []

        try:
            # Track objects in this frame
            results = self.video_predictor.track(frame)

            for obj in results:
                track_id = obj.get('id', len(detections))
                mask = obj.get('mask')
                bbox = obj.get('bbox')
                score = obj.get('score', 0.8)

                if score < self.confidence_threshold:
                    continue

                # Convert mask to bbox if needed
                if bbox is None and mask is not None:
                    bbox = self._mask_to_bbox(mask)

                if bbox is not None:
                    # Check aspect ratio (plates are wider than tall)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / height if height > 0 else 0

                    if aspect_ratio < 1.3:
                        continue  # Likely not a plate

                    detections.append({
                        'track_id': track_id,
                        'bbox': bbox,
                        'mask': mask,
                        'confidence': score,
                        'frame_idx': self.frame_index
                    })

                    # Store in tracked objects
                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = {
                            'bboxes': [], 'masks': [], 'frames': []
                        }
                    self.tracked_objects[track_id]['bboxes'].append(bbox)
                    self.tracked_objects[track_id]['masks'].append(mask)
                    self.tracked_objects[track_id]['frames'].append(self.frame_index)

        except Exception as e:
            logger.error(f"SAM3 video tracking error: {e}")

        return detections

    def _detect_hf_mode(self, frame: Image.Image) -> List[Dict]:
        """Detection using Hugging Face SAM3."""
        detections = []

        try:
            inputs = self.processor(
                images=frame,
                text=self.text_prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process outputs
            masks = self.processor.post_process_masks(
                outputs.pred_masks,
                inputs.original_sizes,
                inputs.reshaped_input_sizes
            )

            scores = outputs.iou_scores

            for i, (mask, score) in enumerate(zip(masks[0], scores[0])):
                if score < self.confidence_threshold:
                    continue

                bbox = self._mask_to_bbox(mask.cpu().numpy())

                if bbox is not None:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / height if height > 0 else 0

                    if aspect_ratio < 1.3:
                        continue

                    detections.append({
                        'track_id': i,
                        'bbox': bbox,
                        'mask': mask.cpu().numpy(),
                        'confidence': float(score),
                        'frame_idx': self.frame_index
                    })

        except Exception as e:
            logger.error(f"HF SAM3 detection error: {e}")

        return detections

    def _detect_image_mode(self, frame: Image.Image) -> List[Dict]:
        """Detection using SAM3 image model (per-frame)."""
        detections = []

        try:
            # Use text prompt for detection
            results = self.processor(
                image=frame,
                text=self.text_prompt
            )

            for i, result in enumerate(results):
                mask = result.get('mask')
                score = result.get('score', 0.8)

                if score < self.confidence_threshold:
                    continue

                bbox = self._mask_to_bbox(mask) if mask is not None else None

                if bbox is not None:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / height if height > 0 else 0

                    if aspect_ratio < 1.3:
                        continue

                    detections.append({
                        'bbox': bbox,
                        'mask': mask,
                        'confidence': score,
                        'frame_idx': self.frame_index
                    })

        except Exception as e:
            logger.error(f"SAM3 image detection error: {e}")

        return detections

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[List[int]]:
        """Convert binary mask to bounding box."""
        if mask is None:
            return None

        try:
            if len(mask.shape) > 2:
                mask = mask.squeeze()

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not rows.any() or not cols.any():
                return None

            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            return [int(x1), int(y1), int(x2), int(y2)]
        except:
            return None

    def get_tracked_plates(self) -> Dict:
        """Get all tracked plate objects."""
        return self.tracked_objects

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None and self.device == "cuda":
            del self.model
            torch.cuda.empty_cache()
