# Setup Guide

## For Qwen Face Blur

```bash
pip install transformers torch pillow opencv-python
```

Then run:
```bash
python qwen_face_blur.py image.jpg
```

## For SAM3 Multi-Category

```bash
# Install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Authenticate
huggingface-cli login
```

Request access: https://huggingface.co/facebook/sam3

Then run:
```bash
python sam3_detect.py image.jpg --config categories_config.json
```

## Usage Examples

### Qwen (Recommended)
```bash
python qwen_face_blur.py peoples.jpg output.jpg 71
```

### SAM3
```bash
python sam3_detect.py road.jpg --blur face --detect pothole crack
```
