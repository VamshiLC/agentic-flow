# Face Blurring Solutions

## Quick Start

### Option 1: Qwen2.5-VL-7B (Recommended - Fastest)

```bash
python qwen_face_blur.py image.jpg
```

- Uses your existing Qwen model
- 2-3x faster than SAM3
- No extra VRAM needed

### Option 2: SAM3 Multi-Category

```bash
# Blur faces only
python sam3_detect.py image.jpg --blur face

# Blur faces + detect defects
python sam3_detect.py image.jpg \
  --blur face \
  --detect pothole crack "license plate"

# Use config file
python sam3_detect.py image.jpg --config categories_config.json
```

## Configuration

Edit `categories_config.json`:
```json
{
  "blur_categories": ["face"],
  "detect_categories": ["license plate", "pothole", "crack", "car"]
}
```

## Files

- `qwen_face_blur.py` - Qwen + Gaussian blur (fastest)
- `sam3_detect.py` - SAM3 multi-category detection
- `categories_config.json` - Detection configuration
