# Quick Start Guide

## Two Approaches Available

### 🚀 **Simple Approach (Recommended for Getting Started)**

**No vLLM server needed** - Works immediately!

Based on your existing `detector.py` pattern. Loads Qwen3-VL directly.

#### Installation
```bash
pip install torch torchvision transformers accelerate pillow opencv-python numpy tqdm
```

#### Usage
```bash
# Single image
python main_simple.py --mode image --input gopro_frame.jpg

# Video (1 frame per second)
python main_simple.py --mode video --input gopro_video.mp4 --fps 1.0

# Custom categories
python main_simple.py --mode image --input frame.jpg \
    --categories potholes alligator_cracks abandoned_vehicle
```

**Pros:**
- ✅ Simple setup - no server required
- ✅ Works immediately
- ✅ Uses your proven detection patterns
- ✅ Based on your working `detector.py`

**Cons:**
- ⚠️ Slower inference (no vLLM optimization)
- ⚠️ Loads model in each process
- ⚠️ ~8GB VRAM for Qwen3-VL-4B

---

### ⚡ **Advanced Approach (Faster for Production)**

**Requires vLLM server** - Optimized for batch processing.

Based on Meta's official SAM3 Agent pattern.

#### Setup
1. **Install SAM3:**
   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3 && pip install -e .
   cd ..
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup vLLM server:**
   ```bash
   # Create separate conda env
   conda create -n vllm python=3.12
   conda activate vllm
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

   # Start server
   vllm serve Qwen/Qwen3-VL-4B-Instruct \
       --tensor-parallel-size 1 \
       --allowed-local-media-path / \
       --enforce-eager \
       --port 8001
   ```

4. **Run detection (in original env):**
   ```bash
   python main.py --mode video --input gopro_video.mp4
   ```

**Pros:**
- ✅ Much faster inference (vLLM optimization)
- ✅ Better for batch processing
- ✅ Follows Meta's official pattern
- ✅ Can scale with tensor parallelism

**Cons:**
- ⚠️ More complex setup
- ⚠️ Requires vLLM server running
- ⚠️ Need separate conda environment

---

## Quick Comparison

| Feature | Simple (main_simple.py) | Advanced (main.py) |
|---------|-------------------------|---------------------|
| Setup complexity | Easy | Complex |
| vLLM server | Not needed | Required |
| Inference speed | Slower | Faster |
| Memory usage | ~8GB VRAM | ~12GB VRAM (both models) |
| Best for | Testing, small batches | Production, large batches |
| Based on | Your detector.py | Meta SAM3 Agent |

---

## Detection Categories

Both approaches detect these categories:

### Road Defects
- **potholes** - Severe holes in pavement
- **alligator_cracks** - Web-like cracking
- **longitudinal_cracks** - Parallel to traffic
- **transverse_cracks** - Perpendicular to traffic
- **road_surface_damage** - General deterioration

### Social Issues (from your detector.py)
- **abandoned_vehicle** - Abandoned cars
- **homeless_encampment** - Tents/shelters
- **homeless_person** - People living on streets

### Infrastructure
- **manholes** - Utility access points
- **damaged_paint** - Faded road markings
- **damaged_crosswalks** - Faded crosswalks
- **dumped_trash** - Debris
- **street_signs** - Traffic signs
- **traffic_lights** - Signal lights
- **tyre_marks** - Tire marks

---

## Output Format

Both approaches produce the same JSON output:

```json
{
  "detections": [
    {
      "label": "pothole",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "color": [0, 0, 255]
    }
  ],
  "num_detections": 1,
  "text_response": "..."
}
```

---

## Which Approach Should I Use?

### Use **Simple Approach** (`main_simple.py`) if:
- 👍 You want to get started quickly
- 👍 Processing small batches (< 100 frames)
- 👍 Don't want to manage vLLM server
- 👍 Testing/prototyping
- 👍 Want code similar to your existing `detector.py`

### Use **Advanced Approach** (`main.py`) if:
- 👍 Processing large batches (100+ frames)
- 👍 Need faster inference
- 👍 Have production deployment
- 👍 Can manage vLLM server
- 👍 Want to use SAM3 Agent features

---

## Tips

1. **Start with Simple:** Get familiar with the system using `main_simple.py`
2. **Move to Advanced:** Once you need speed, migrate to `main.py` with vLLM
3. **GPU Memory:** If running out of VRAM, use `--device cpu` (slow but works)
4. **Categories:** Limit to specific categories to improve accuracy:
   ```bash
   python main_simple.py --mode image --input frame.jpg \
       --categories potholes alligator_cracks
   ```

---

## Troubleshooting

### Simple Mode Errors

**"CUDA out of memory"**
```bash
# Use CPU (slower but works)
python main_simple.py --mode image --input frame.jpg --device cpu
```

**"Model not found"**
```bash
# Login to Hugging Face
huggingface-cli login
```

### Advanced Mode Errors

**"vLLM server not running"**
```bash
# Check server
curl http://localhost:8001/health

# Restart server
pkill -f 'vllm serve'
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8001 ...
```

**"SAM3 not found"**
```bash
# Install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

---

## Next Steps

1. **Test Simple Mode:**
   ```bash
   python main_simple.py --mode image --input test_image.jpg
   ```

2. **Process Your Videos:**
   ```bash
   python main_simple.py --mode video --input gopro_video.mp4 --fps 1.0
   ```

3. **Review Output:**
   - Check `output_simple/` directory
   - View annotated images/videos
   - Read JSON detection files

4. **Customize:**
   - Edit `detector_unified.py` to adjust prompts
   - Modify `INFRASTRUCTURE_CATEGORIES` in `detector_unified.py`
   - Tune confidence thresholds
