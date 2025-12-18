"""
Face Detection and Blurring Utilities

Privacy-preserving face blurring for video frames before infrastructure detection.
Supports multiple backends: MediaPipe (recommended), OpenCV Haar Cascades (fallback).
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Literal


# Try to import MediaPipe
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# Try to import RetinaFace
try:
    from retinaface import RetinaFace as RF
    HAS_RETINAFACE = True
except ImportError:
    HAS_RETINAFACE = False

# Try to import SAM3 (for all-in-one solution)
try:
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False

# Show warnings if advanced backends not available
if not HAS_RETINAFACE and not HAS_MEDIAPIPE and not HAS_SAM3:
    print("WARNING: No advanced face detection backends available. Using OpenCV Haar Cascades fallback.")
    print("For better face detection, install:")
    print("  - SAM3 (all-in-one): Already in your environment!")
    print("  - RetinaFace (best accuracy): pip install retinaface")
    print("  - MediaPipe (fast): pip install mediapipe")


class FaceBlurrer:
    """
    Face detection and blurring with multiple backend support.

    Backends:
        - 'sam3': SAM3 text prompts (all-in-one, precise masks) [UNIFIED]
        - 'retinaface': RetinaFace (state-of-the-art, best accuracy) [BEST]
        - 'mediapipe': Google MediaPipe (fast, accurate, CPU-friendly) [GOOD]
        - 'opencv': OpenCV Haar Cascades (lightweight fallback) [BASIC]

    Example:
        blurrer = FaceBlurrer(backend='sam3', blur_type='gaussian', sam3_model=model)
        blurred_frame = blurrer.blur_faces(frame)
    """

    def __init__(
        self,
        backend: Literal['sam3', 'retinaface', 'mediapipe', 'opencv'] = 'retinaface',
        blur_type: Literal['gaussian', 'pixelate'] = 'gaussian',
        blur_strength: int = 51,
        min_detection_confidence: float = 0.5,
        expand_bbox_ratio: float = 0.2,
        sam3_model=None,
        sam3_processor=None,
        device: str = 'cuda'
    ):
        """
        Initialize face blurrer.

        Args:
            backend: Detection backend ('sam3', 'retinaface', 'mediapipe', or 'opencv')
            blur_type: Blur method ('gaussian' or 'pixelate')
            blur_strength: Blur kernel size for gaussian (must be odd) or pixel size for pixelate
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            expand_bbox_ratio: Expand bounding box by this ratio to ensure full face coverage
            sam3_model: SAM3 model instance (required if backend='sam3')
            sam3_processor: SAM3 processor instance (required if backend='sam3')
            device: Device for SAM3 ('cuda' or 'cpu')
        """
        self.backend = backend
        self.blur_type = blur_type
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.min_detection_confidence = min_detection_confidence
        self.expand_bbox_ratio = expand_bbox_ratio
        self.device = device

        # SAM3-specific
        self.sam3_model = sam3_model
        self.sam3_processor = sam3_processor

        # Initialize backend with fallback chain
        if backend == 'sam3':
            if not HAS_SAM3:
                print("SAM3/Transformers not available. Falling back to RetinaFace...")
                if HAS_RETINAFACE:
                    self.backend = 'retinaface'
                    self._init_retinaface()
                elif HAS_MEDIAPIPE:
                    print("RetinaFace not available. Falling back to MediaPipe...")
                    self.backend = 'mediapipe'
                    self._init_mediapipe()
                else:
                    print("MediaPipe not available. Falling back to OpenCV Haar Cascades.")
                    self.backend = 'opencv'
                    self._init_opencv()
            else:
                self._init_sam3()
        elif backend == 'retinaface':
            if not HAS_RETINAFACE:
                print("RetinaFace not available. Falling back to MediaPipe...")
                if HAS_MEDIAPIPE:
                    self.backend = 'mediapipe'
                    self._init_mediapipe()
                else:
                    print("MediaPipe not available. Falling back to OpenCV Haar Cascades.")
                    self.backend = 'opencv'
                    self._init_opencv()
            else:
                self._init_retinaface()
        elif backend == 'mediapipe':
            if not HAS_MEDIAPIPE:
                print("MediaPipe not available. Falling back to OpenCV Haar Cascades.")
                self.backend = 'opencv'
                self._init_opencv()
            else:
                self._init_mediapipe()
        elif backend == 'opencv':
            self._init_opencv()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'sam3', 'retinaface', 'mediapipe', or 'opencv'")

    def _init_sam3(self):
        """Initialize SAM3 face detection."""
        if self.sam3_model is None or self.sam3_processor is None:
            print("WARNING: SAM3 model/processor not provided. You must pass sam3_model and sam3_processor.")
            print("Attempting to load SAM3...")
            try:
                from models.sam3_text_prompt_loader import load_sam3_text_prompt_model
                self.sam3_model, self.sam3_processor, _ = load_sam3_text_prompt_model(device=self.device)
                print(f"Loaded SAM3 model on {self.device}")
            except Exception as e:
                print(f"ERROR loading SAM3: {e}")
                print("Falling back to OpenCV...")
                self.backend = 'opencv'
                self._init_opencv()
                return

        print(f"Initialized SAM3 face detection (confidence >= {self.min_detection_confidence})")
        print(f"  Device: {self.device}")
        print(f"  Note: SAM3 is slower but provides precise masks")

    def _init_retinaface(self):
        """Initialize RetinaFace detection."""
        # RetinaFace doesn't need explicit initialization - model loads on first use
        print(f"Initialized RetinaFace detection (confidence >= {self.min_detection_confidence})")

    def _init_mediapipe(self):
        """Initialize MediaPipe face detection."""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for close-range, 1 for full-range
            min_detection_confidence=self.min_detection_confidence
        )
        print(f"Initialized MediaPipe face detection (confidence >= {self.min_detection_confidence})")

    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade face detection."""
        # Load pre-trained Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

        print(f"Initialized OpenCV Haar Cascade face detection")

    def detect_faces_sam3(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using SAM3 text prompts.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        from PIL import Image as PILImage

        h, w = image.shape[:2]

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        try:
            # Prepare inputs with text prompt "face"
            inputs = self.sam3_processor(
                images=pil_image,
                text_prompts=["face"],
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.sam3_model(**inputs)

            # Extract masks and scores
            masks = outputs.pred_masks[0]  # [num_masks, H, W]
            scores = outputs.iou_scores[0]  # [num_masks]

            bboxes = []

            # Process each detected mask
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Filter by confidence
                if score.item() < self.min_detection_confidence:
                    continue

                # Convert mask to numpy
                mask_np = mask.cpu().numpy()

                # Resize mask to original image size if needed
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                # Get bounding box from mask
                mask_binary = (mask_np > 0.5).astype(np.uint8)

                # Find contours
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    continue

                # Get largest contour
                contour = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(contour)

                # Expand bounding box
                expand_w = int(w_box * self.expand_bbox_ratio)
                expand_h = int(h_box * self.expand_bbox_ratio)

                x1 = max(0, x - expand_w)
                y1 = max(0, y - expand_h)
                x2 = min(w, x + w_box + expand_w)
                y2 = min(h, y + h_box + expand_h)

                bboxes.append((x1, y1, x2, y2))

            return bboxes

        except Exception as e:
            print(f"SAM3 face detection error: {e}")
            return []

    def detect_faces_retinaface(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using RetinaFace.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        h, w = image.shape[:2]

        # Run RetinaFace detection
        try:
            detections = RF.detect_faces(image)
        except Exception:
            # No faces detected or error
            return []

        bboxes = []
        if detections:
            for key, detection in detections.items():
                # Get facial area (bounding box)
                facial_area = detection.get('facial_area', None)
                confidence = detection.get('score', 0.0)

                # Filter by confidence
                if facial_area and confidence >= self.min_detection_confidence:
                    x1, y1, x2, y2 = facial_area

                    # Expand bounding box slightly
                    width = x2 - x1
                    height = y2 - y1
                    expand_w = int(width * self.expand_bbox_ratio)
                    expand_h = int(height * self.expand_bbox_ratio)

                    x1 = max(0, x1 - expand_w)
                    y1 = max(0, y1 - expand_h)
                    x2 = min(w, x2 + expand_w)
                    y2 = min(h, y2 + expand_h)

                    bboxes.append((x1, y1, x2, y2))

        return bboxes

    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MediaPipe.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Detect faces
        results = self.face_detector.process(image_rgb)

        bboxes = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box in relative coordinates
                bbox = detection.location_data.relative_bounding_box

                # Convert to absolute coordinates
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Expand bounding box slightly to ensure full face coverage
                width = x2 - x1
                height = y2 - y1
                expand_w = int(width * self.expand_bbox_ratio)
                expand_h = int(height * self.expand_bbox_ratio)

                x1 = max(0, x1 - expand_w)
                y1 = max(0, y1 - expand_h)
                x2 = min(w, x2 + expand_w)
                y2 = min(h, y2 + expand_h)

                bboxes.append((x1, y1, x2, y2))

        return bboxes

    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV Haar Cascades.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        bboxes = []
        for (x, y, w_box, h_box) in faces:
            # Expand bounding box
            expand_w = int(w_box * self.expand_bbox_ratio)
            expand_h = int(h_box * self.expand_bbox_ratio)

            x1 = max(0, x - expand_w)
            y1 = max(0, y - expand_h)
            x2 = min(w, x + w_box + expand_w)
            y2 = min(h, y + h_box + expand_h)

            bboxes.append((x1, y1, x2, y2))

        return bboxes

    def apply_gaussian_blur(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply Gaussian blur to a region.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Image with blurred region
        """
        x1, y1, x2, y2 = bbox

        # Extract face region
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return image

        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (self.blur_strength, self.blur_strength), 0)

        # Replace face region with blurred version
        image[y1:y2, x1:x2] = blurred_face

        return image

    def apply_pixelation(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply pixelation to a region.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Image with pixelated region
        """
        x1, y1, x2, y2 = bbox

        # Extract face region
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return image

        # Get original dimensions
        h, w = face_region.shape[:2]

        # Determine pixel block size
        pixel_size = self.blur_strength

        # Resize down then up for pixelation effect
        temp = cv2.resize(face_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        # Replace face region with pixelated version
        image[y1:y2, x1:x2] = pixelated_face

        return image

    def blur_faces(self, image: np.ndarray, return_face_count: bool = False) -> np.ndarray:
        """
        Detect and blur all faces in an image.

        Args:
            image: Input image (BGR format from cv2.imread or video frame)
            return_face_count: If True, return (blurred_image, num_faces)

        Returns:
            Blurred image, or (blurred_image, num_faces) if return_face_count=True
        """
        # Make a copy to avoid modifying original
        output = image.copy()

        # Detect faces
        if self.backend == 'sam3':
            bboxes = self.detect_faces_sam3(image)
        elif self.backend == 'retinaface':
            bboxes = self.detect_faces_retinaface(image)
        elif self.backend == 'mediapipe':
            bboxes = self.detect_faces_mediapipe(image)
        else:
            bboxes = self.detect_faces_opencv(image)

        # Apply blur to each face
        for bbox in bboxes:
            if self.blur_type == 'gaussian':
                output = self.apply_gaussian_blur(output, bbox)
            elif self.blur_type == 'pixelate':
                output = self.apply_pixelation(output, bbox)

        if return_face_count:
            return output, len(bboxes)
        return output

    def __del__(self):
        """Cleanup resources."""
        if self.backend == 'mediapipe' and hasattr(self, 'face_detector'):
            self.face_detector.close()


def blur_faces_in_frame(
    frame: np.ndarray,
    backend: str = 'retinaface',
    blur_type: str = 'gaussian',
    blur_strength: int = 51
) -> np.ndarray:
    """
    Convenience function to blur faces in a single frame.

    Args:
        frame: Input frame (BGR format)
        backend: 'retinaface', 'mediapipe', or 'opencv'
        blur_type: 'gaussian' or 'pixelate'
        blur_strength: Blur intensity (kernel size for gaussian, pixel size for pixelate)

    Returns:
        Frame with blurred faces

    Example:
        frame = cv2.imread("frame.jpg")
        blurred = blur_faces_in_frame(frame, backend='retinaface')
        cv2.imwrite("blurred_frame.jpg", blurred)
    """
    blurrer = FaceBlurrer(backend=backend, blur_type=blur_type, blur_strength=blur_strength)
    return blurrer.blur_faces(frame)


# For backward compatibility and simple usage
def blur_faces(image: np.ndarray, blur_type: str = 'gaussian', strength: int = 51) -> np.ndarray:
    """
    Simple face blurring function (auto-selects best available backend).

    Args:
        image: Input image (BGR)
        blur_type: 'gaussian' or 'pixelate'
        strength: Blur strength

    Returns:
        Image with blurred faces
    """
    # Auto-select best available backend
    if HAS_RETINAFACE:
        backend = 'retinaface'
    elif HAS_MEDIAPIPE:
        backend = 'mediapipe'
    else:
        backend = 'opencv'
    return blur_faces_in_frame(image, backend=backend, blur_type=blur_type, blur_strength=strength)
