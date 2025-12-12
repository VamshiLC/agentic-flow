# ASH Infrastructure Detection - Usage Guide

This guide explains the two ways to run infrastructure detection and which one to use.

---

## Quick Start: Which One Should I Use?

### Use `main_simple.py` (Hugging Face) if:
- ✅ You want the simplest setup (no server required)
- ✅ Processing single images or short videos
- ✅ First time using the system
- ✅ Don't want to deal with server management

### Use `main.py` (vLLM Server) if:
- ✅ Processing many videos in production
- ✅ Want absolute fastest inference
- ✅ Already have vLLM server running
- ✅ Need concurrent processing

---

## Option 1: Hugging Face Direct (Recommended for Most Users)

### Installation

```bash
# Basic installation
pip install torch transformers pillow opencv-python tqdm

# Optional: For 8-bit quantization (saves ~50% memory)
pip install bitsandbytes
```

### Basic Usage

```bash
# Process a single image
python main_simple.py --mode image --input frame.jpg

# Process a video at 2 FPS
python main_simple.py --mode video --input video.mp4 --fps 2.0
```

### Advanced Features

#### 1. Batch Processing (Faster!)
Process multiple frames at once for better GPU utilization:

```bash
# Process 4 frames at a time (2-3x faster)
python main_simple.py --mode video --input video.mp4 --fps 2.0 --batch-size 4

# Larger batches for powerful GPUs
python main_simple.py --mode video --input video.mp4 --fps 2.0 --batch-size 8
```

#### 2. Memory Optimization
Reduce memory usage with quantization:

```bash
# Use 8-bit quantization (requires bitsandbytes)
python main_simple.py --mode video --input video.mp4 --quantize

# Combine with low memory mode
python main_simple.py --mode video --input video.mp4 --quantize --low-memory
```

#### 3. Faster Processing (Skip Video Output)
Generate only JSON detections without annotated video:

```bash
python main_simple.py --mode video --input video.mp4 --no-video
```

#### 4. Specific Categories Only
Detect only critical issues:

```bash
python main_simple.py --mode image --input frame.jpg \
  --categories potholes alligator_cracks abandoned_vehicles
```

#### 5. CPU Mode
Force CPU usage (useful for testing):

```bash
python main_simple.py --mode image --input frame.jpg --device cpu
```

### Memory Requirements

| Mode | VRAM Needed | Notes |
|------|-------------|-------|
| Standard (FP16) | ~8GB | Default on GPU |
| Quantized (8-bit) | ~4GB | With `--quantize` flag |
| CPU | N/A | Slower but works on any machine |

### Performance Benchmarks

On a single RTX 3090 (24GB):
- **Sequential**: ~2 seconds per frame
- **Batch size 4**: ~0.6 seconds per frame (3.3x faster!)
- **Batch size 8**: ~0.4 seconds per frame (5x faster!)

---

## Option 2: vLLM Server (Advanced)

### Installation

```bash
pip install vllm

# Install SAM3 (if using agent mode)
# Follow SAM3 installation instructions
```

### Setup

1. **Start vLLM server** (in a separate terminal):

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --tensor-parallel-size 1 \
  --allowed-local-media-path / \
  --enforce-eager \
  --port 8001
```

2. **Check server is running**:

```bash
python main.py --check-server
```

### Usage

```bash
# Process a single image
python main.py --mode image --input frame.jpg

# Process a video
python main.py --mode video --input video.mp4 --sample-rate 15

# With video chunking for large files
python main.py --mode video --input long_video.mp4 \
  --enable-chunking \
  --chunk-duration 600
```

### When to Use vLLM

- **Production deployments** with continuous processing
- **Multiple concurrent requests** (API server)
- **Very large batch processing** (100+ videos)
- When you need **absolute maximum throughput**

---

## Comparison Table

| Feature | main_simple.py (Hugging Face) | main.py (vLLM) |
|---------|-------------------------------|----------------|
| **Setup Complexity** | Simple | Complex (server required) |
| **Speed (single)** | ~2s/frame | ~1s/frame |
| **Speed (batched)** | ~0.5s/frame | ~0.5s/frame |
| **Memory Usage** | 8GB (4GB with quantization) | 8GB |
| **Production Ready** | ✅ Yes | ✅ Yes |
| **Batch Processing** | ✅ Built-in | ⚠️ Manual |
| **Server Management** | ❌ Not needed | ✅ Required |
| **Quantization** | ✅ Yes | ❌ No |
| **Best For** | Most use cases | High-volume production |

---

## Output Format

Both methods produce the same JSON output:

```json
{
  "frame_id": "frame_000001.jpg",
  "num_detections": 2,
  "detections": [
    {
      "label": "potholes",
      "category": "potholes",
      "bbox": [100, 200, 400, 500],
      "confidence": 0.8,
      "color": [0, 0, 255]
    },
    {
      "label": "alligator_cracks",
      "category": "alligator_cracks",
      "bbox": [50, 300, 600, 700],
      "confidence": 0.8,
      "color": [0, 200, 255]
    }
  ],
  "response": "Full text response from model"
}
```

---

## Troubleshooting

### Out of Memory Errors

```bash
# Try these in order:
# 1. Enable quantization
python main_simple.py --mode video --input video.mp4 --quantize

# 2. Add low memory mode
python main_simple.py --mode video --input video.mp4 --quantize --low-memory

# 3. Reduce batch size
python main_simple.py --mode video --input video.mp4 --batch-size 2

# 4. Use CPU (slowest but works)
python main_simple.py --mode video --input video.mp4 --device cpu
```

### Slow Processing

```bash
# Increase batch size (if memory allows)
python main_simple.py --mode video --input video.mp4 --batch-size 8

# Skip video output (2x faster)
python main_simple.py --mode video --input video.mp4 --no-video

# Reduce FPS (process fewer frames)
python main_simple.py --mode video --input video.mp4 --fps 1.0
```

### Model Loading Errors

```bash
# Make sure you have enough disk space
df -h

# Clear Hugging Face cache if needed
rm -rf ~/.cache/huggingface/

# Re-download model
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')"
```

---

## Complete Examples

### Example 1: Quick Test on Single Image
```bash
python main_simple.py --mode image --input test_frame.jpg
```

### Example 2: Process 10-minute GoPro Video (Optimized)
```bash
python main_simple.py \
  --mode video \
  --input gopro_video.mp4 \
  --fps 2.0 \
  --batch-size 4 \
  --output results/
```

### Example 3: Low Memory System (8GB GPU)
```bash
python main_simple.py \
  --mode video \
  --input video.mp4 \
  --fps 1.0 \
  --batch-size 2 \
  --quantize \
  --low-memory
```

### Example 4: Maximum Speed (No Video Output)
```bash
python main_simple.py \
  --mode video \
  --input video.mp4 \
  --fps 2.0 \
  --batch-size 8 \
  --no-video
```

### Example 5: Critical Issues Only
```bash
python main_simple.py \
  --mode video \
  --input video.mp4 \
  --fps 2.0 \
  --categories potholes alligator_cracks
```

---

## Recommendation

**For 95% of users**: Start with `main_simple.py`. It's simpler, has all the features you need, and performs just as well with batching enabled.

**Only use `main.py` (vLLM) if**: You're building a production API server or processing 100+ videos per day.

---

## Need Help?

- Check the main README.md for architecture details
- See QUICKSTART.md for installation instructions
- Open an issue on GitHub for problems
