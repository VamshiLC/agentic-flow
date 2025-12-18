# SAM3 Face Blur Setup

## Installation

```bash
# Create environment
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone and install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Install dependencies
pip install opencv-python pillow numpy

# Authenticate with Hugging Face (required for model access)
huggingface-cli login
```

**Note:** You must request access to SAM3 checkpoints at https://huggingface.co/facebook/sam3

## Usage

```bash
# Basic usage
python sam3_face_blur.py peoples1.png

# Specify output file
python sam3_face_blur.py peoples1.png result.jpg

# Adjust blur strength (default: 51)
python sam3_face_blur.py peoples1.png result.jpg 71
```

## How It Works

1. **Load SAM3** - Uses official Meta SAM3 model
2. **Detect faces** - Text prompt "face" finds all faces
3. **Get bounding boxes** - Extracts box coordinates from SAM3 output
4. **Blur** - Applies Gaussian blur to face regions
5. **Save** - Outputs blurred image

That's it. Clean and simple.

## Sources

- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 Hugging Face](https://huggingface.co/facebook/sam3)
- [Meta AI SAM3](https://ai.meta.com/sam3/)
