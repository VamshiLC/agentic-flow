#!/bin/bash
#
# Setup script for ASH Infrastructure Detection Agent on AWS SageMaker
#
# This script installs all dependencies and starts the vLLM server
# for running the detection agent on SageMaker.
#

set -e  # Exit on error

echo "=========================================="
echo "ASH Infrastructure Detection Agent Setup"
echo "=========================================="
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ WARNING: CUDA not detected. Agent will run on CPU (very slow)"
fi

echo ""
echo "STEP 1: Installing SAM3 from source..."
echo "=========================================="

# Clone SAM3 repository
if [ -d "/tmp/sam3" ]; then
    echo "SAM3 repository already exists, pulling latest changes..."
    cd /tmp/sam3
    git pull
else
    echo "Cloning SAM3 repository..."
    git clone https://github.com/facebookresearch/sam3.git /tmp/sam3
    cd /tmp/sam3
fi

# Install SAM3
echo "Installing SAM3..."
pip install -e .

echo "✓ SAM3 installed successfully"

echo ""
echo "STEP 2: Installing Python dependencies..."
echo "=========================================="

# Go back to sam3-agent directory
cd -

# Install requirements
pip install -r requirements.txt

echo "✓ Python dependencies installed"

echo ""
echo "STEP 3: Setting up vLLM (for Qwen3-VL)..."
echo "=========================================="

# Check if vLLM conda env exists
if conda env list | grep -q "vllm"; then
    echo "vLLM conda environment already exists"
    VLLM_EXISTS=1
else
    echo "Creating vLLM conda environment..."
    conda create -n vllm python=3.12 -y
    VLLM_EXISTS=0
fi

# Activate vLLM environment and install vLLM
echo "Activating vLLM environment and installing vLLM..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm

# Install vLLM with CUDA support
if [ "$VLLM_EXISTS" -eq 0 ]; then
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
    echo "✓ vLLM installed successfully"
else
    echo "✓ vLLM environment ready"
fi

# Deactivate conda env
conda deactivate

echo ""
echo "STEP 4: Starting vLLM server..."
echo "=========================================="

# Kill existing vLLM server if running
if pgrep -f "vllm serve" > /dev/null; then
    echo "Stopping existing vLLM server..."
    pkill -f "vllm serve"
    sleep 2
fi

# Start vLLM server in background
echo "Starting Qwen3-VL-4B-Instruct server on port 8001..."

# Activate vLLM environment and start server
conda activate vllm

nohup vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --tensor-parallel-size 1 \
    --allowed-local-media-path / \
    --enforce-eager \
    --port 8001 \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to start
echo "Waiting for vLLM server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠ WARNING: vLLM server may not have started. Check vllm_server.log"
        echo "You can manually check with: curl http://localhost:8001/health"
    fi
    sleep 2
    echo -n "."
done
echo ""

conda deactivate

echo ""
echo "STEP 5: Validating installation..."
echo "=========================================="

# Test imports
echo "Testing Python imports..."
python3 << EOF
import torch
import sam3
from transformers import AutoProcessor
print("✓ All imports successful")
print(f"  - PyTorch version: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "The ASH Infrastructure Detection Agent is ready to use."
echo ""
echo "Usage examples:"
echo ""
echo "  # Process a single image"
echo "  python main.py --mode image --input frame.jpg --output ./results"
echo ""
echo "  # Process a video (1 frame per second)"
echo "  python main.py --mode video --input video.mp4 --output ./results --sample-rate 30"
echo ""
echo "  # Check vLLM server status"
echo "  python main.py --check-server"
echo ""
echo "  # List available models"
echo "  python main.py --list-models"
echo ""
echo "Server logs: vllm_server.log"
echo "To stop vLLM server: pkill -f 'vllm serve'"
echo ""
