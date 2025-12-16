# SAM3-Only Infrastructure Detection

SAM3-only detection system that uses **text prompts directly** with SAM3 for both detection and segmentation, without the Qwen vision-language model.

## Architecture Comparison

### Full Agentic Flow (main_simple.py)
```
Qwen2.5-VL (detects) → SAM3 (segments with boxes)
```
- **Pros**: More accurate detection, better semantic understanding
- **Cons**: Requires both models (~12GB VRAM), slower

### SAM3-Only Flow (main_sam3_only.py)
```
SAM3 (detects + segments with text prompts)
```
- **Pros**: Single model, lower memory (~6GB VRAM), faster
- **Cons**: Less accurate detection, simpler semantic understanding

## Installation

### 1. Install Dependencies

```bash
# Install SAM3-only requirements
pip install -r requirements_sam3_only.txt

# OR install from main requirements
pip install -r requirements.txt
```

### 2. Login to Hugging Face

```bash
# Required for SAM3 model access
huggingface-cli login
```

### 3. Accept Model Terms

Visit https://huggingface.co/facebook/sam3 and accept the model terms.

## Usage

### Single Image Detection

```bash
# Detect all categories
python main_sam3_only.py --mode image --input path/to/image.jpg

# Detect specific categories only
python main_sam3_only.py --mode image --input image.jpg \
    --categories potholes alligator_cracks longitudinal_cracks

# Lower confidence threshold (more detections)
python main_sam3_only.py --mode image --input image.jpg --confidence 0.2

# Skip visualization (JSON only)
python main_sam3_only.py --mode image --input image.jpg --no-viz
```

### Video Detection

```bash
# Process video at original FPS
python main_sam3_only.py --mode video --input path/to/video.mp4

# Process at specific FPS (2 FPS recommended for road videos)
python main_sam3_only.py --mode video --input video.mp4 --fps 2.0

# Limit to first 50 frames (for testing)
python main_sam3_only.py --mode video --input video.mp4 --max-frames 50

# Specific categories with custom output directory
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes damaged_crosswalks manholes \
    --output-dir results/sam3_detections
```

### List Available Categories

```bash
python main_sam3_only.py --list-categories
```

## Output Structure

```
output_sam3_only/
├── image_name_sam3_detections.json      # Detection results
├── image_name_sam3_annotated.jpg        # Visualized image
├── video_name_sam3_detections.json      # Video detection results
└── video_name_sam3_annotated.mp4        # Annotated video
```

### JSON Format

```json
{
  "video_path": "path/to/video.mp4",
  "video_name": "video",
  "detection_time": "2025-12-16T...",
  "total_frames": 100,
  "fps": 2.0,
  "num_detections": 15,
  "categories_searched": ["potholes", "alligator_cracks", ...],
  "detections": [
    {
      "frame_number": 5,
      "timestamp": 2.5,
      "label": "potholes",
      "category": "potholes",
      "prompt_used": "pothole",
      "confidence": 0.85,
      "bbox": [100, 200, 400, 500],
      "has_mask": true,
      "object_id": 1,
      "tracking_id": "0_1"
    }
  ]
}
```

## Detection Categories

The system can detect 15 infrastructure categories:

### Critical Issues
- **potholes** - Holes in road surface
- **alligator_cracks** - Interconnected web-like cracks

### Road Damage
- **transverse_cracks** - Cracks across road width
- **longitudinal_cracks** - Cracks along road length
- **damaged_crosswalks** - Faded/damaged crosswalk markings
- **damaged_paint** - Deteriorated road markings

### Social Issues
- **homeless_encampment** - Tents/temporary shelters
- **homeless_person** - People living on streets
- **abandoned_vehicle** - Derelict vehicles
- **dumped_trash** - Illegally dumped waste

### Infrastructure
- **manholes** - Utility access covers
- **street_signs** - Traffic/regulatory signs
- **traffic_lights** - Signal lights

### Minor Issues
- **tyre_marks** - Tire marks on pavement
- **graffiti** - Spray paint vandalism

## Performance

### Speed (RTX 3090, 24GB VRAM)
- **Single image**: ~15-20 seconds (processes all 15 categories)
- **Video (2 FPS)**: ~15-20 seconds per frame
- **Categories**: ~1-2 seconds per category per frame

### Memory Usage
- **GPU Memory**: ~5-6 GB VRAM
- **CPU Memory**: ~4-8 GB RAM

### Tips for Faster Processing

1. **Use specific categories** instead of all 15:
   ```bash
   python main_sam3_only.py --mode video --input video.mp4 \
       --categories potholes alligator_cracks  # Only 2 categories = faster
   ```

2. **Reduce FPS** for videos:
   ```bash
   python main_sam3_only.py --mode video --input video.mp4 --fps 1.0
   ```

3. **Skip visualization** if you only need detection data:
   ```bash
   python main_sam3_only.py --mode video --input video.mp4 --no-viz
   ```

## Comparison with Full Agentic System

| Feature | SAM3-Only | Full Agentic (Qwen + SAM3) |
|---------|-----------|---------------------------|
| **Detection Method** | Text prompts | Vision-language model |
| **Accuracy** | Good | Excellent |
| **Speed** | Faster (~1-2s/category) | Slower (~2-3s total) |
| **VRAM Usage** | ~6 GB | ~12 GB |
| **False Positives** | More | Fewer |
| **Best For** | Quick scans, limited resources | Production, high accuracy |

## When to Use SAM3-Only

✅ **Use SAM3-Only When:**
- You need quick results
- Limited GPU memory (<12 GB)
- Testing/prototyping
- Specific known categories
- Lower accuracy is acceptable

❌ **Use Full Agentic System When:**
- Need highest accuracy
- Production deployment
- Complex scenes
- Sufficient GPU memory (12+ GB)
- Reducing false positives is critical

## Troubleshooting

### Out of Memory Error
```bash
# Reduce categories
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes alligator_cracks

# Reduce frames
python main_sam3_only.py --mode video --input video.mp4 --max-frames 20
```

### Model Access Error
```bash
# Login to Hugging Face
huggingface-cli login

# Accept terms at: https://huggingface.co/facebook/sam3
```

### Slow CPU Performance
```bash
# Force CUDA usage
python main_sam3_only.py --mode image --input image.jpg --device cuda
```

## Examples

### Example 1: Quick Pothole Detection
```bash
python main_sam3_only.py --mode image --input road.jpg \
    --categories potholes --confidence 0.25
```

### Example 2: Multi-Category Video Analysis
```bash
python main_sam3_only.py --mode video --input dashcam.mp4 \
    --categories potholes alligator_cracks damaged_crosswalks \
    --fps 2.0 --output-dir results/dashcam_analysis
```

### Example 3: Test Run (First 10 Frames)
```bash
python main_sam3_only.py --mode video --input test.mp4 \
    --max-frames 10 --confidence 0.3
```

## Files Created

New files for SAM3-only system (existing code unchanged):

- `models/sam3_text_prompt_loader.py` - SAM3 model loader
- `inference/sam3_only_single_frame.py` - Image processing
- `inference/sam3_only_video.py` - Video processing
- `main_sam3_only.py` - Main entry point
- `requirements_sam3_only.txt` - Dependencies
- `README_SAM3_ONLY.md` - This file

## Notes

- SAM3 uses **text prompts** for detection, which is less semantically aware than Qwen
- Each category is processed **independently**, which can be slower for many categories
- For production use with highest accuracy, use the full agentic system (`main_simple.py`)
- SAM3-only is great for **specific use cases** where you know exactly what to look for
