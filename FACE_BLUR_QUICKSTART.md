# Face Blurring Quick Start Guide

## Installation

### Install RetinaFace (Recommended - Best Accuracy)

```bash
pip install retinaface
```

**Why RetinaFace?**
- State-of-the-art face detection accuracy
- Works with challenging angles, occlusion, and lighting
- Better than MediaPipe and OpenCV for production use
- Automatically falls back to MediaPipe or OpenCV if not installed

### Alternative: Install MediaPipe (Good Balance)

```bash
pip install mediapipe
```

### Built-in Fallback: OpenCV

OpenCV Haar Cascades are already included in your environment (no additional installation needed).

---

## Usage Examples

### 1. Process Video with Face Blurring (RetinaFace)

```bash
python main_simple.py \
    --mode video \
    --input gopro_video.mp4 \
    --fps 2.0 \
    --enable-face-blur
```

This will:
- Extract frames at 2 FPS
- Detect faces using RetinaFace (default)
- Apply Gaussian blur to detected faces
- Run infrastructure detection on blurred frames

### 2. Process Video with Different Backends

```bash
# Using RetinaFace (best)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-backend retinaface

# Using MediaPipe (good)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-backend mediapipe

# Using OpenCV (basic)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-backend opencv
```

### 3. Different Blur Types

```bash
# Gaussian blur (default - smooth effect)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-type gaussian

# Pixelation (mosaic/censored effect)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-type pixelate
```

### 4. Adjust Blur Strength

```bash
# Stronger blur (higher value = more blur)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-strength 71

# Maximum blur
python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur --face-blur-strength 101
```

### 5. Process Single Image

```bash
python main_simple.py --mode image --input frame.jpg --enable-face-blur
```

---

## Complete Example (Production Settings)

```bash
python main_simple.py \
    --mode video \
    --input gopro_highway_inspection.mp4 \
    --output results/ \
    --fps 2.0 \
    --batch-size 4 \
    --enable-face-blur \
    --face-blur-backend retinaface \
    --face-blur-type gaussian \
    --face-blur-strength 71
```

**Output:**
```
ASH INFRASTRUCTURE DETECTION - AGENTIC PIPELINE (Qwen3-VL + SAM3)
======================================================================
Model: Qwen/Qwen2.5-VL-7B-Instruct
Device: auto-detect
Pipeline: Qwen3-VL detection + SAM3 segmentation
Face blurring: Enabled

Processing video: gopro_highway_inspection.mp4
  Video FPS: 30.00
  Resolution: 1920x1080
  Total frames: 54000
  Target processing FPS: 2.0
  Face blurring: ENABLED
    Backend: retinaface, Type: gaussian

Initialized RetinaFace detection (confidence >= 0.5)

Extracting frames...
Extracted 1800 frames to process
  Total faces blurred: 23

Processing 1800 frames...
[Detection progress...]

Summary:
  Total detections: 145
  Frames with detections: 98/1800
```

---

## Testing Face Detection

Test face blurring on a sample image before processing full videos:

```bash
python test_face_blur.py test_image.jpg
```

This creates comparison outputs in `test_output/`:
- `test_retinaface_gaussian.jpg` - RetinaFace + Gaussian
- `test_mediapipe_gaussian.jpg` - MediaPipe + Gaussian
- `test_mediapipe_pixelate.jpg` - MediaPipe + Pixelation
- `test_opencv_gaussian.jpg` - OpenCV + Gaussian
- `test_auto_backend.jpg` - Auto-selected backend

---

## Backend Comparison

| Backend | Accuracy | Speed | Requirements | Best For |
|---------|----------|-------|--------------|----------|
| **RetinaFace** | ★★★★★ | ★★★☆☆ | `pip install retinaface` | Production, challenging conditions |
| **MediaPipe** | ★★★★☆ | ★★★★★ | `pip install mediapipe` | Balanced performance |
| **OpenCV** | ★★☆☆☆ | ★★★★★ | Built-in | Quick testing, frontal faces |

---

## Using Face Blurring in Code

### Python API Example

```python
from utils.face_blur import FaceBlurrer
import cv2

# Initialize with RetinaFace
blurrer = FaceBlurrer(
    backend='retinaface',
    blur_type='gaussian',
    blur_strength=51,
    min_detection_confidence=0.5
)

# Process image
image = cv2.imread("frame.jpg")
blurred_image, num_faces = blurrer.blur_faces(image, return_face_count=True)

print(f"Detected and blurred {num_faces} faces")
cv2.imwrite("blurred_frame.jpg", blurred_image)
```

### Convenience Function

```python
from utils.face_blur import blur_faces_in_frame
import cv2

# Auto-selects RetinaFace (if installed)
image = cv2.imread("frame.jpg")
blurred = blur_faces_in_frame(image, backend='retinaface')
cv2.imwrite("blurred.jpg", blurred)
```

---

## Command-Line Options Reference

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--enable-face-blur` | flag | disabled | Enable face blurring |
| `--face-blur-backend` | `retinaface`, `mediapipe`, `opencv` | `retinaface` | Detection method |
| `--face-blur-type` | `gaussian`, `pixelate` | `gaussian` | Blur style |
| `--face-blur-strength` | integer (odd for gaussian) | 51 | Blur intensity |

---

## Troubleshooting

### RetinaFace Not Found
```
ERROR: Could not import RetinaFace
```
**Solution:** `pip install retinaface`

### RetinaFace Falls Back to MediaPipe
```
RetinaFace not available. Falling back to MediaPipe...
```
**Solution:** This is normal if RetinaFace isn't installed. Install with `pip install retinaface` for best results.

### Low Detection Rate
- RetinaFace has the highest detection rate (~95%+)
- MediaPipe is good (~90%+)
- OpenCV is basic (~70-85%, frontal faces only)
- Try RetinaFace for better results: `--face-blur-backend retinaface`

### Blur Too Weak
```bash
# Increase blur strength (must be odd for gaussian)
--face-blur-strength 71   # Strong
--face-blur-strength 101  # Very strong
```

---

## Performance

### Speed (per frame on CPU)
- **RetinaFace**: ~50-100ms (slightly slower but worth it for accuracy)
- **MediaPipe**: ~10-30ms
- **OpenCV**: ~5-15ms

### Accuracy
- **RetinaFace**: ~95%+ detection rate, handles difficult angles/lighting
- **MediaPipe**: ~90%+ detection rate, good for most cases
- **OpenCV**: ~70-85% detection rate, frontal faces only

### Memory
- All backends: ~50-200MB RAM
- No GPU memory required (runs on CPU)

---

## Why Use Face Blurring?

1. **Privacy Compliance**: GDPR, CCPA, and other regulations
2. **Public Data**: Required for publishing infrastructure datasets
3. **Municipal Projects**: Government/city projects often require face anonymization
4. **Best Practice**: Industry standard for road/infrastructure monitoring

---

## Next Steps

1. **Install RetinaFace**: `pip install retinaface`
2. **Test on sample image**: `python test_face_blur.py test_image.jpg`
3. **Process video**: `python main_simple.py --mode video --input video.mp4 --fps 2.0 --enable-face-blur`
4. **Review results**: Check `output/` directory for blurred frames and detections

For detailed documentation, see: `Documentation/FACE_BLURRING_GUIDE.md`
