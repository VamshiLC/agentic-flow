# Testing SAM3 Face Blurring on Images

## Three Ways to Test

### Method 1: Dedicated Test Script (Recommended for Testing)

Test SAM3 face blurring only (no infrastructure detection):

```bash
python test_sam3_face_blur.py your_image.jpg
```

**Output:**
```
test_sam3_face_output/
├── original.jpg              # Original image
├── sam3_gaussian_blur.jpg    # Gaussian blur (medium)
├── sam3_pixelate.jpg         # Pixelation effect
└── sam3_strong_blur.jpg      # Strong gaussian blur
```

**What it does:**
- Loads SAM3 model
- Tests 3 different blur settings
- Shows detection count
- Fast comparison

---

### Method 2: Full Pipeline (Infrastructure + Face Blur)

Use the main SAM3-only pipeline for both face blurring and infrastructure detection:

```bash
python main_sam3_only.py \
    --mode image \
    --input your_image.jpg \
    --enable-face-blur
```

**Output:**
```
output_sam3_only/
├── potholes/
│   ├── frames/                # Blurred original image
│   ├── annotated/             # Annotated + blurred
│   └── detections.json
├── alligator_cracks/
│   └── ...
└── [other categories]/
```

**What it does:**
- Blurs faces first
- Detects infrastructure (potholes, cracks, etc.)
- Saves organized by category

---

### Method 3: Python API (For Custom Integration)

```python
import cv2
import torch
from utils.face_blur import FaceBlurrer
from models.sam3_text_prompt_loader import load_sam3_text_prompt_model

# Load SAM3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, processor, _ = load_sam3_text_prompt_model(device=device)

# Create blurrer
blurrer = FaceBlurrer(
    backend='sam3',
    blur_type='gaussian',
    blur_strength=51,
    sam3_model=model,
    sam3_processor=processor,
    device=device
)

# Process image
image = cv2.imread("your_image.jpg")
blurred, num_faces = blurrer.blur_faces(image, return_face_count=True)

print(f"Detected {num_faces} faces")
cv2.imwrite("blurred_output.jpg", blurred)
```

---

## Quick Test Examples

### 1. Just Test Face Blurring

```bash
# Test SAM3 face blur on single image
python test_sam3_face_blur.py test_image.jpg
```

**Expected output:**
```
======================================================================
SAM3 FACE BLURRING TEST
======================================================================
Image: test_image.jpg

Image size: 1920x1080

1. Loading SAM3 model...
   Device: cuda
   ✓ SAM3 model loaded

2. Testing SAM3 + Gaussian blur...
   Processing...
   ✓ Detected 3 face(s)
   ✓ Saved: test_sam3_face_output/sam3_gaussian_blur.jpg

3. Testing SAM3 + Pixelation...
   Processing...
   ✓ Detected 3 face(s)
   ✓ Saved: test_sam3_face_output/sam3_pixelate.jpg

4. Testing SAM3 + Strong blur...
   Processing...
   ✓ Detected 3 face(s)
   ✓ Saved: test_sam3_face_output/sam3_strong_blur.jpg

======================================================================
TEST COMPLETE
======================================================================

All outputs saved to: test_sam3_face_output/
```

---

### 2. Full Pipeline on Image

```bash
# Face blur + infrastructure detection
python main_sam3_only.py \
    --mode image \
    --input image.jpg \
    --enable-face-blur \
    --face-blur-type gaussian \
    --face-blur-strength 71
```

---

### 3. Test Multiple Blur Strengths

```bash
# Weak blur
python main_sam3_only.py --mode image --input image.jpg --enable-face-blur --face-blur-strength 31

# Medium blur (default)
python main_sam3_only.py --mode image --input image.jpg --enable-face-blur --face-blur-strength 51

# Strong blur
python main_sam3_only.py --mode image --input image.jpg --enable-face-blur --face-blur-strength 71

# Maximum blur
python main_sam3_only.py --mode image --input image.jpg --enable-face-blur --face-blur-strength 101
```

---

### 4. Test Pixelation

```bash
python main_sam3_only.py \
    --mode image \
    --input image.jpg \
    --enable-face-blur \
    --face-blur-type pixelate \
    --face-blur-strength 20
```

---

## Performance Expectations

### Speed (per image)
- **Loading SAM3**: ~3-5 seconds (one-time)
- **Face detection**: ~2-3 seconds per image
- **Infrastructure detection**: ~1-2 seconds per category
- **Total**: ~5-10 seconds for face blur + infrastructure

### Accuracy
- **Face detection**: ~60-80% (depends on angles, lighting)
- **Infrastructure detection**: ~85-95%

---

## Troubleshooting

### No faces detected

```
✓ Detected 0 face(s)
```

**Possible reasons:**
1. No faces in image
2. Faces too small
3. Unusual angles
4. Poor lighting

**Solutions:**
- Verify faces are visible in image
- Try lower confidence (in Python API)
- Check image quality

---

### SAM3 not loading

```
ERROR loading SAM3: No module named 'transformers'
```

**Solution:**
```bash
pip install transformers torch
```

---

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead (slower)
export CUDA_VISIBLE_DEVICES=""
python test_sam3_face_blur.py image.jpg
```

---

### Very slow processing

**Expected:** SAM3 is slower than specialized face detectors.
- SAM3: ~2-3 seconds per image
- RetinaFace: ~0.05-0.1 seconds per image

**For faster processing**, use hybrid approach:
```bash
python main_simple.py --mode image --input image.jpg --enable-face-blur --face-blur-backend retinaface
```

---

## Comparison: Test All Backends

To compare SAM3 with other face detection backends:

```bash
# Test all backends (SAM3, RetinaFace, MediaPipe, OpenCV)
python test_face_blur.py test_image.jpg
```

This generates comparisons in `test_output/`:
- `test_retinaface_gaussian.jpg`
- `test_mediapipe_gaussian.jpg`
- `test_opencv_gaussian.jpg`
- `test_auto_backend.jpg`

Then test SAM3 separately:
```bash
python test_sam3_face_blur.py test_image.jpg
```

Compare in `test_sam3_face_output/`:
- `sam3_gaussian_blur.jpg`
- `sam3_pixelate.jpg`
- `sam3_strong_blur.jpg`

---

## Visual Comparison

### Gaussian Blur (blur_strength=51)
- Smooth, natural-looking blur
- Good for general privacy
- Recommended for most cases

### Pixelation (blur_strength=20)
- Mosaic/censored effect
- More visible privacy indicator
- Lower values = larger pixel blocks

### Strong Blur (blur_strength=101)
- Maximum blur
- Good for compliance documentation
- Ensures complete face obscuration

---

## Command Reference

### Test Scripts

```bash
# SAM3-only test (recommended)
python test_sam3_face_blur.py <image>

# All backends test (comparison)
python test_face_blur.py <image>
```

### Full Pipeline

```bash
# Basic
python main_sam3_only.py --mode image --input <image> --enable-face-blur

# With options
python main_sam3_only.py \
    --mode image \
    --input <image> \
    --enable-face-blur \
    --face-blur-type <gaussian|pixelate> \
    --face-blur-strength <31-101> \
    --categories <categories>
```

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install transformers torch opencv-python`
- [ ] Have a test image with faces
- [ ] Run test script: `python test_sam3_face_blur.py test_image.jpg`
- [ ] Check output in `test_sam3_face_output/`
- [ ] Compare blur types (gaussian vs pixelate)
- [ ] Try full pipeline: `python main_sam3_only.py --mode image --input test_image.jpg --enable-face-blur`

---

## Summary

**Fastest way to test:**
```bash
python test_sam3_face_blur.py your_image.jpg
```

**Full pipeline test:**
```bash
python main_sam3_only.py --mode image --input your_image.jpg --enable-face-blur
```

**Check results:**
```bash
ls test_sam3_face_output/
ls output_sam3_only/
```

Done! 🎉
