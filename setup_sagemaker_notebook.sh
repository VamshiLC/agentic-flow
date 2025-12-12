#!/bin/bash
#
# Quick Setup for ASH Infrastructure Detection in SageMaker Notebook Instance
# Optimized for Hugging Face Transformers (NO vLLM server needed!)
#

set -e  # Exit on error

echo "========================================================================="
echo "   ASH Infrastructure Detection - SageMaker Notebook Setup"
echo "   Using Hugging Face Transformers (No vLLM Server Required!)"
echo "========================================================================="
echo ""

# ===== STEP 1: Check GPU =====
echo "STEP 1: Checking GPU availability..."
echo "---------------------------------------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected!"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠ WARNING: No CUDA/GPU detected. Will run on CPU (very slow)"
    echo "  For testing, you should use a GPU instance like ml.g4dn.xlarge"
    echo ""
fi

# ===== STEP 2: Hugging Face Token Setup =====
echo ""
echo "STEP 2: Hugging Face Token Setup (REQUIRED)"
echo "---------------------------------------------------------------------"
echo "The models require a Hugging Face access token to download."
echo ""

# Check if already logged in
if huggingface-cli whoami &> /dev/null; then
    echo "✓ Already logged in to Hugging Face"
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -n 1)
    echo "  User: $HF_USER"
    echo ""
else
    echo "You need a Hugging Face token. Get one here:"
    echo "  👉 https://huggingface.co/settings/tokens"
    echo ""
    echo "Steps:"
    echo "  1. Click 'New token'"
    echo "  2. Name: 'sagemaker-testing'"
    echo "  3. Type: 'Read'"
    echo "  4. Copy the token (starts with hf_...)"
    echo ""

    # Try to get token from environment first
    if [ -z "$HF_TOKEN" ]; then
        read -sp "Paste your Hugging Face token here: " HF_TOKEN
        echo ""
    else
        echo "Using HF_TOKEN from environment"
    fi

    # Login using the token
    echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

    if [ $? -eq 0 ]; then
        echo "✓ Successfully logged in to Hugging Face!"
        echo ""
    else
        echo "✗ Failed to login. Please check your token and try again."
        exit 1
    fi
fi

# ===== STEP 3: Install Dependencies =====
echo ""
echo "STEP 3: Installing Python dependencies..."
echo "---------------------------------------------------------------------"

# Install required packages
echo "Installing core dependencies (this may take 2-3 minutes)..."
pip install -q --upgrade pip
pip install -q torch>=2.0.0 torchvision>=0.15.0
pip install -q transformers>=4.40.0 accelerate>=0.20.0
pip install -q pillow>=10.0.0 opencv-python>=4.8.0
pip install -q tqdm>=4.66.0 numpy>=1.24.0
pip install -q einops>=0.7.0 timm>=0.9.0

echo "✓ Core dependencies installed"
echo ""

# Optional: Install bitsandbytes for quantization
if command -v nvidia-smi &> /dev/null; then
    echo "Installing bitsandbytes for 8-bit quantization support (optional)..."
    pip install -q bitsandbytes>=0.41.0 || echo "⚠ bitsandbytes installation failed (optional, skipping)"
    echo ""
fi

# ===== STEP 4: Validate Installation =====
echo ""
echo "STEP 4: Validating installation..."
echo "---------------------------------------------------------------------"

python3 << 'EOF'
import sys
import torch
from transformers import AutoProcessor

print("Checking imports...")
print(f"  ✓ Python version: {sys.version.split()[0]}")
print(f"  ✓ PyTorch version: {torch.__version__}")
print(f"  ✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ⚠ Running on CPU - inference will be slow")

print("\nAll imports successful!")
EOF

if [ $? -ne 0 ]; then
    echo "✗ Validation failed"
    exit 1
fi

echo ""

# ===== STEP 5: Test Model Loading =====
echo ""
echo "STEP 5: Testing model access (downloading if needed)..."
echo "---------------------------------------------------------------------"
echo "This will download Qwen3-VL-4B-Instruct (~8GB) on first run."
echo "Subsequent runs will use cached model."
echo ""

python3 << 'EOF'
from transformers import AutoProcessor
import os

print("Testing model access...")
try:
    # Test if we can access the model
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        trust_remote_code=True
    )
    print("✓ Model access confirmed!")
    print("✓ Model will be downloaded on first inference run")
except Exception as e:
    print(f"✗ Error accessing model: {e}")
    print("\nTroubleshooting:")
    print("  1. Check your HF token is valid")
    print("  2. Visit https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct")
    print("  3. Click 'Agree and access repository' if prompted")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# ===== DONE =====
echo ""
echo "========================================================================="
echo "                        ✓ SETUP COMPLETE!"
echo "========================================================================="
echo ""
echo "You're ready to process videos with batch processing!"
echo ""
echo "Quick Start:"
echo "---------------------------------------------------------------------"
echo ""
echo "  # Test with a single image"
echo "  python main_simple.py --mode image --input test-frame.jpg"
echo ""
echo "  # Process video WITHOUT batching (baseline)"
echo "  python main_simple.py --mode video --input video.mp4 --fps 2.0"
echo ""
echo "  # Process video WITH batching (2-5x faster!)"
echo "  python main_simple.py --mode video --input video.mp4 --fps 2.0 --batch-size 4"
echo ""
echo "  # Low memory mode (for 8GB GPUs)"
echo "  python main_simple.py --mode video --input video.mp4 --quantize --batch-size 2"
echo ""
echo "Recommended batch sizes for ml.g4dn.xlarge (16GB GPU):"
echo "  - Standard: --batch-size 2-4"
echo "  - Quantized (--quantize): --batch-size 4-6"
echo ""
echo "Performance expectations:"
echo "  - Sequential: ~2s per frame"
echo "  - Batch size 4: ~0.6s per frame (3x faster!)"
echo ""
echo "For full documentation, see: SAGEMAKER_QUICKSTART.md"
echo ""
echo "========================================================================="
echo ""
