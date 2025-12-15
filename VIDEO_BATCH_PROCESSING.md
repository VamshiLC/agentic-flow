# Video Batch Processing Guide

## Quick Start

### Process Video with All Categories
```bash
# Recommended settings for ml.g5.xlarge (24GB VRAM)
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --output ./output \
    --fps 2.0 \
    --batch-size 4
```

## Batch Size Recommendations

### For Qwen2.5-VL-7B on ml.g5.xlarge (24GB VRAM)

| Batch Size | VRAM Usage | Speed | Stability | Use Case |
|------------|------------|-------|-----------|----------|
| 1 | ~18GB | Slow | ✅ Very Safe | Initial testing |
| 2 | ~19GB | Medium | ✅ Safe | Balanced |
| 4 | ~20-21GB | Fast | ✅ Recommended | **Best choice** ⭐ |
| 8 | ~22-23GB | Faster | ⚠️ Tight | May OOM |
| 16 | ~24GB+ | Fastest | ❌ OOM Risk | Too large |

**Recommended: batch-size=4** for good balance of speed and stability

## Processing Modes

### Mode 1: All Categories, Single Output
**Script:** `main_simple.py`

```bash
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --output results/ \
    --fps 2.0 \
    --batch-size 4
```

**Output:**
```
results/
└── video/
    ├── frames/              # Extracted frames
    ├── annotated_frames/    # Frames with detections
    ├── video_detected.mp4   # Annotated video
    └── video_detections.json # All detections
```

**Best for:** Quick analysis, viewing annotated video

### Mode 2: Multi-Category Organization
**Script:** `process_multi_category.py`

```bash
python process_multi_category.py \
    --input video.mp4 \
    --output results/ \
    --fps 2.0 \
    --batch-size 1  # Note: Currently sequential
```

**Output:**
```
results/
├── potholes/
│   ├── frames/           # Original frames with potholes
│   ├── annotated/        # Annotated frames
│   └── detections.json   # Pothole-specific data
├── homeless_encampment/
│   ├── frames/
│   ├── annotated/
│   └── detections.json
├── ... (one folder per category)
└── summary.json          # Overall summary
```

**Best for:** Category-specific analysis, organized results

## Complete Command Examples

### Example 1: Test Short Video (Fast)
```bash
python main_simple.py \
    --mode video \
    --input test_video.mp4 \
    --output test_output/ \
    --fps 1.0 \
    --batch-size 4 \
    --quantize
```
- 1 FPS (fewer frames)
- Batch size 4
- 8-bit quantization for speed

### Example 2: Full Video Processing (Balanced)
```bash
python main_simple.py \
    --mode video \
    --input full_video.mp4 \
    --output full_output/ \
    --fps 2.0 \
    --batch-size 4
```
- 2 FPS (good coverage)
- Batch size 4 (recommended)
- Full precision

### Example 3: High-Quality Processing (Slow)
```bash
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --output hq_output/ \
    --fps 5.0 \
    --batch-size 2
```
- 5 FPS (dense sampling)
- Smaller batch for stability
- Maximum quality

### Example 4: Multi-Category Organization
```bash
python process_multi_category.py \
    --input video.mp4 \
    --output organized_results/ \
    --fps 2.0 \
    --device cuda \
    --quantize \
    --low-memory
```
- Organized by category
- Memory optimizations enabled

### Example 5: Exclude Problematic Categories
```bash
# Note: You'll need to modify the script to support --exclude
# Or use Python directly:

python -c "
from detector_unified import get_detector
from PIL import Image
import cv2

detector = get_detector(
    categories=None,
    exclude_categories=['graffiti', 'tyre_marks', 'manholes']
)

# Process video frames...
"
```

## Performance Estimates

### For 1-minute video (30 fps = 1800 frames)

| FPS | Frames Processed | Batch Size | Time Estimate |
|-----|------------------|------------|---------------|
| 1.0 | 60 | 1 | ~2-3 minutes |
| 1.0 | 60 | 4 | ~1 minute ⭐ |
| 2.0 | 120 | 1 | ~4-5 minutes |
| 2.0 | 120 | 4 | ~2 minutes ⭐ |
| 5.0 | 300 | 1 | ~10-12 minutes |
| 5.0 | 300 | 4 | ~5-6 minutes |

**⭐ Recommended:** 2.0 FPS with batch_size=4 (~2 minutes for 1-min video)

## Advanced Options

### Quantization (Faster, Less Memory)
```bash
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --quantize \
    --low-memory \
    --batch-size 8  # Can use larger batch with quantization
```

**Trade-off:**
- ✅ 40-50% faster processing
- ✅ Uses ~12-14GB instead of 18-20GB
- ⚠️ Slight accuracy decrease (~5%)

### Skip Video Output (Faster)
```bash
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --no-video \
    --batch-size 4
```
- Only saves JSON detections
- Faster (no video encoding)
- Good for data collection

### Custom Categories Only
```bash
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --categories potholes alligator_cracks homeless_encampment \
    --batch-size 4
```
- Only detect specific categories
- Faster processing
- Focused results

## Monitoring GPU Usage

While processing, monitor GPU in another terminal:

```bash
watch -n 1 nvidia-smi
```

Look for:
- **GPU Memory Used**: Should stay under 22GB for stability
- **GPU Utilization**: Should be 90-100% during processing
- **Temperature**: Should stay under 80°C

## Troubleshooting

### Out of Memory (OOM) Error
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 2`
2. Enable quantization: `--quantize`
3. Enable low memory: `--low-memory`
4. Reduce FPS: `--fps 1.0`

### Processing Too Slow

**Solutions:**
1. Increase batch size: `--batch-size 8` (watch memory!)
2. Reduce FPS: `--fps 1.0`
3. Enable quantization: `--quantize`
4. Skip video output: `--no-video`

### Too Many False Positives

**Solutions:**
1. Use filters (already enabled)
2. Exclude categories: Modify code to add `exclude_categories=['graffiti', 'manholes']`
3. Increase confidence thresholds in `detection_filters.py`

## Output Analysis

### Check JSON Results
```bash
# View summary
cat results/video/video_detections.json | jq '.processed_frames'

# Count detections per category
cat results/video/video_detections.json | jq '[.frame_detections[].detections[].label] | group_by(.) | map({category: .[0], count: length})'

# Find frames with most detections
cat results/video/video_detections.json | jq '.frame_detections | sort_by(.num_detections) | reverse | .[0:5]'
```

### View Annotated Video
```bash
# Open with default player
xdg-open results/video/video_detected.mp4

# Or use ffplay
ffplay results/video/video_detected.mp4
```

## Recommended Workflow

### Step 1: Quick Test (30 seconds)
```bash
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --output quick_test/ \
    --fps 1.0 \
    --batch-size 4 \
    --quantize
```

### Step 2: Check Results
```bash
# View annotated video
xdg-open quick_test/video/video_detected.mp4

# Check JSON for false positives
cat quick_test/video/video_detections.json | jq '.frame_detections[].detections[] | select(.label == "graffiti")'
```

### Step 3: Full Processing
```bash
# If test looks good, run full processing
python main_simple.py \
    --mode video \
    --input video.mp4 \
    --output final_results/ \
    --fps 2.0 \
    --batch-size 4
```

## Best Practices

1. ✅ **Start small**: Test with 1 FPS and small batch first
2. ✅ **Monitor GPU**: Watch nvidia-smi during processing
3. ✅ **Save incrementally**: Process long videos in chunks
4. ✅ **Check quality**: Review first few results before full processing
5. ✅ **Use filters**: Keep smart filters enabled (already default)

## Complete Example Session

```bash
# 1. Test on short clip
python main_simple.py --mode video --input test.mp4 --fps 1.0 --batch-size 4

# 2. Check if results look good
ls -lh output/test/

# 3. Process full video with optimal settings
python main_simple.py --mode video --input full_video.mp4 --fps 2.0 --batch-size 4 --output results/

# 4. While processing, monitor GPU
watch -n 1 nvidia-smi

# 5. When done, analyze results
cat results/full_video/full_video_detections.json | jq '.processed_frames, .frame_detections | length'
```

That's it! You're ready for batch video processing with all categories! 🎥
