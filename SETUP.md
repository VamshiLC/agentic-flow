# Setup Guide

## Qwen Full Pipeline (Recommended)

### Requirements
```bash
pip install transformers torch pillow opencv-python numpy
```

### Usage
```bash
python qwen_full_pipeline.py road.jpg
```

**Output:** Face blur + infrastructure detection in `qwen_results/`

---

## SAM3 (Optional)

Only needed if using `sam3_detect.py`:

```bash
# Install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Authenticate
huggingface-cli login
```

Request access: https://huggingface.co/facebook/sam3

---

## Quick Test

```bash
# Full pipeline (face blur + defect detection)
python qwen_full_pipeline.py your_image.jpg

# Face blur only
python qwen_face_blur.py your_image.jpg

# SAM3 (if installed)
python sam3_detect.py your_image.jpg --blur face --detect pothole
```
