# Face Blurring Guide

Automatically detect and blur faces in video frames for privacy protection before infrastructure detection.

## Overview

Face blurring is integrated into the pipeline as a **preprocessing step** that runs **before** infrastructure detection. This ensures:
- Privacy protection for people in GoPro footage
- Compliant with data privacy regulations (GDPR, etc.)
- No impact on infrastructure detection accuracy (faces are not road defects!)

## Installation

### Option 1: MediaPipe (Recommended)

MediaPipe provides the best accuracy and works well with various angles and lighting conditions.

```bash
pip install mediapipe
```

### Option 2: OpenCV Only (Lightweight)

If you can't install MediaPipe, the system will automatically fall back to OpenCV's Haar Cascades (already included).

## Usage

### Quick Start

#### Single Image
```bash
# Basic usage (MediaPipe + Gaussian blur)
python main_simple.py --mode image --input frame.jpg --enable-face-blur

# With pixelation instead of blur
python main_simple.py --mode image --input frame.jpg --enable-face-blur --face-blur-type pixelate
```

#### Video Processing
```bash
# Process video with face blurring
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur

# With custom blur settings
python main_simple.py --mode video --input video.mp4 \
    --fps 2.0 \
    --enable-face-blur \
    --face-blur-backend mediapipe \
    --face-blur-type gaussian \
    --face-blur-strength 71
```

### All Face Blurring Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--enable-face-blur` | flag | disabled | Enable face blurring |
| `--face-blur-backend` | `mediapipe`, `opencv` | `mediapipe` | Face detection method |
| `--face-blur-type` | `gaussian`, `pixelate` | `gaussian` | Blur style |
| `--face-blur-strength` | integer (odd for gaussian) | 51 | Blur intensity |

### Backend Comparison

#### MediaPipe
- **Pros**: More accurate, handles varied angles/lighting, CPU-friendly
- **Cons**: Requires additional dependency
- **Best for**: Production use, diverse video conditions
- **Installation**: `pip install mediapipe`

#### OpenCV Haar Cascades
- **Pros**: No extra dependencies, lightweight, fast
- **Cons**: Less accurate with angles/occlusion, frontal faces only
- **Best for**: Simple cases, quick testing, minimal dependencies

### Blur Type Comparison

#### Gaussian Blur
- Smooth, natural-looking blur
- Good for general privacy protection
- Adjustable strength (higher = more blur)
- Recommended for most use cases

#### Pixelation
- Mosaic/censored effect
- More visible privacy protection
- Lower values = larger pixel blocks
- Good for compliance documentation

## Advanced Usage

### Using the Video Processor API

```python
from inference.video_processor import process_video

results = process_video(
    video_path="gopro_video.mp4",
    output_dir="output/",
    sample_rate=15,  # 2 fps at 30fps video
    enable_face_blur=True,
    face_blur_backend='mediapipe',
    face_blur_type='gaussian',
    face_blur_strength=51
)
```

### Using the Face Blurrer Directly

```python
from utils.face_blur import FaceBlurrer
import cv2

# Initialize blurrer
blurrer = FaceBlurrer(
    backend='mediapipe',
    blur_type='gaussian',
    blur_strength=51,
    min_detection_confidence=0.5,
    expand_bbox_ratio=0.2  # Expand face bbox by 20%
)

# Load and blur image
image = cv2.imread("frame.jpg")
blurred_image, num_faces = blurrer.blur_faces(image, return_face_count=True)
print(f"Detected and blurred {num_faces} faces")

# Save result
cv2.imwrite("blurred_frame.jpg", blurred_image)
```

### Standalone Face Blurring Utility

```python
from utils.face_blur import blur_faces_in_frame
import cv2

# Auto-selects best available backend
image = cv2.imread("frame.jpg")
blurred = blur_faces_in_frame(image, blur_type='gaussian', blur_strength=51)
cv2.imwrite("blurred.jpg", blurred)
```

## Testing

Test face blurring on a sample image:

```bash
python test_face_blur.py test_image.jpg
```

This generates comparison outputs with different backends and blur types in `test_output/`:
- `test_mediapipe_gaussian.jpg` - MediaPipe + Gaussian blur
- `test_mediapipe_pixelate.jpg` - MediaPipe + Pixelation
- `test_opencv_gaussian.jpg` - OpenCV + Gaussian blur
- `test_auto_backend.jpg` - Auto-selected backend

## Performance Considerations

### Speed
- **MediaPipe**: ~10-30ms per frame (CPU)
- **OpenCV Haar Cascade**: ~5-15ms per frame (CPU)
- Minimal impact on overall pipeline (infrastructure detection is the bottleneck)

### Memory
- Negligible memory overhead (~50-100MB for models)
- No additional GPU memory required (runs on CPU)

### Accuracy
- **MediaPipe**: Detects ~95%+ of visible faces in typical conditions
- **OpenCV**: Detects ~70-85% (frontal faces only)

## Configuration Tips

### For Maximum Privacy Protection
```bash
--enable-face-blur \
--face-blur-backend mediapipe \
--face-blur-type gaussian \
--face-blur-strength 101  # Very strong blur
```

### For Balanced Performance/Accuracy
```bash
--enable-face-blur \
--face-blur-backend mediapipe \
--face-blur-type gaussian \
--face-blur-strength 51  # Default, good balance
```

### For Lightweight Processing
```bash
--enable-face-blur \
--face-blur-backend opencv \
--face-blur-type gaussian \
--face-blur-strength 31  # Lighter blur
```

## Integration with Infrastructure Detection

Face blurring happens in **STEP 2** (Frame Extraction), before infrastructure detection:

```
Video Processing Pipeline:
┌─────────────────────────────────┐
│ STEP 1: Video Analysis          │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│ STEP 2: Frame Extraction        │
│   - Extract frames              │
│   - ✓ BLUR FACES (if enabled)   │  ← Face blurring happens here
│   - Save frames                 │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│ STEP 3: Load Detector           │
│   (Qwen + SAM3)                 │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│ STEP 4: Infrastructure Detection│
│   (processes blurred frames)    │
└─────────────────────────────────┘
```

This ensures:
1. **Privacy-first**: Faces blurred before any AI processing
2. **Efficiency**: Blurred frames saved to disk, no re-blurring needed
3. **Transparency**: Saved frames show exactly what was analyzed

## Troubleshooting

### MediaPipe Import Error
```
ERROR: Could not import mediapipe
```
**Solution**: Install MediaPipe with `pip install mediapipe`, or use `--face-blur-backend opencv`

### Haar Cascade Not Found
```
ERROR: Failed to load Haar Cascade classifier
```
**Solution**: Ensure OpenCV is properly installed with `pip install opencv-python`

### Faces Not Detected
- Try MediaPipe instead of OpenCV (`--face-blur-backend mediapipe`)
- Reduce detection confidence: Modify `min_detection_confidence` in code (default: 0.5)
- Check if faces are visible (not occluded, good lighting)

### Blur Too Weak/Strong
- Adjust `--face-blur-strength`:
  - **Gaussian**: Must be odd number (31, 51, 71, 101)
  - **Pixelate**: Lower values = larger pixel blocks (10-30 recommended)

## Examples

### Complete Workflow: GoPro Video with Face Blurring

```bash
# Process 1-hour GoPro video with privacy protection
python main_simple.py \
    --mode video \
    --input gopro_highway_inspection.mp4 \
    --output results/ \
    --fps 2.0 \
    --batch-size 4 \
    --enable-face-blur \
    --face-blur-backend mediapipe \
    --face-blur-type gaussian \
    --face-blur-strength 71
```

**Output:**
```
Video Processing Pipeline
======================================================================
Video: gopro_highway_inspection.mp4
  Video FPS: 30.00
  Resolution: 1920x1080
  Total frames: 108000
  Target processing FPS: 2.0
  Face blurring: ENABLED
    Backend: mediapipe, Type: gaussian

Extracting frames...
Extracted 3600 frames to process
  Total faces blurred: 47

Processing 3600 frames...
[Detection progress...]

Summary:
  Total detections: 234
  Frames with detections: 189/3600
```

## API Reference

### FaceBlurrer Class

```python
class FaceBlurrer:
    def __init__(
        self,
        backend: Literal['mediapipe', 'opencv'] = 'mediapipe',
        blur_type: Literal['gaussian', 'pixelate'] = 'gaussian',
        blur_strength: int = 51,
        min_detection_confidence: float = 0.5,
        expand_bbox_ratio: float = 0.2
    )

    def blur_faces(
        self,
        image: np.ndarray,
        return_face_count: bool = False
    ) -> np.ndarray
```

### Convenience Functions

```python
def blur_faces_in_frame(
    frame: np.ndarray,
    backend: str = 'mediapipe',
    blur_type: str = 'gaussian',
    blur_strength: int = 51
) -> np.ndarray

def blur_faces(
    image: np.ndarray,
    blur_type: str = 'gaussian',
    strength: int = 51
) -> np.ndarray
```

## Best Practices

1. **Always blur faces in production** when processing public road footage
2. **Use MediaPipe** for best accuracy (install with `pip install mediapipe`)
3. **Test on sample images** first to verify face detection works for your use case
4. **Increase blur strength** for compliance documentation (`--face-blur-strength 71` or higher)
5. **Save blurred frames** as proof of privacy protection
6. **Document your settings** in processing logs for audit trails

## Privacy & Compliance

Face blurring helps meet privacy requirements for:
- **GDPR** (EU General Data Protection Regulation)
- **CCPA** (California Consumer Privacy Act)
- **Internal privacy policies** for municipal/government projects
- **Public data collection** regulations

**Note**: This is a technical implementation. Consult legal counsel for compliance requirements specific to your jurisdiction and use case.
