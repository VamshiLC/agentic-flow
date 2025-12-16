# SAM3-Only Quick Start Guide

## 🚀 Installation (One-Time Setup)

```bash
# 1. Install dependencies
pip install -r requirements_sam3_only.txt

# 2. Login to Hugging Face
huggingface-cli login

# 3. Accept SAM3 model terms
# Visit: https://huggingface.co/facebook/sam3
```

## 📸 Image Detection Commands

### Basic Image Detection
```bash
python main_sam3_only.py --mode image --input path/to/image.jpg
```

### Detect Specific Categories
```bash
# Only potholes and cracks
python main_sam3_only.py --mode image --input image.jpg \
    --categories potholes alligator_cracks longitudinal_cracks
```

### Lower Confidence (More Detections)
```bash
python main_sam3_only.py --mode image --input image.jpg --confidence 0.2
```

### Custom Output Directory
```bash
python main_sam3_only.py --mode image --input image.jpg \
    --output-dir results/my_detections
```

## 🎥 Video Detection Commands

### Basic Video Detection
```bash
python main_sam3_only.py --mode video --input path/to/video.mp4 --fps 2.0
```

### Quick Test (First 10 Frames)
```bash
python main_sam3_only.py --mode video --input video.mp4 \
    --max-frames 10 --fps 2.0
```

### Specific Categories Only
```bash
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes damaged_crosswalks manholes \
    --fps 2.0
```

### JSON Only (No Visualization)
```bash
python main_sam3_only.py --mode video --input video.mp4 \
    --fps 2.0 --no-viz
```

## 📋 Common Use Cases

### 1. Quick Pothole Check
```bash
python main_sam3_only.py --mode image --input road_photo.jpg \
    --categories potholes --confidence 0.25
```

### 2. Road Condition Survey (Video)
```bash
python main_sam3_only.py --mode video --input dashcam.mp4 \
    --categories potholes alligator_cracks transverse_cracks longitudinal_cracks \
    --fps 2.0 --output-dir results/road_survey
```

### 3. Social Issues Detection
```bash
python main_sam3_only.py --mode image --input street.jpg \
    --categories homeless_encampment homeless_person abandoned_vehicle dumped_trash
```

### 4. Infrastructure Inventory
```bash
python main_sam3_only.py --mode video --input street_video.mp4 \
    --categories manholes street_signs traffic_lights \
    --fps 1.0
```

### 5. Fast Testing (10 frames, critical issues only)
```bash
python main_sam3_only.py --mode video --input test.mp4 \
    --categories potholes alligator_cracks \
    --max-frames 10 --fps 2.0
```

## 📊 Available Categories

Run this to see all categories:
```bash
python main_sam3_only.py --list-categories
```

### Quick Reference:
- `potholes` - Holes in road surface
- `alligator_cracks` - Interconnected cracks
- `transverse_cracks` - Cracks across road
- `longitudinal_cracks` - Cracks along road
- `damaged_crosswalks` - Faded crosswalks
- `damaged_paint` - Deteriorated road markings
- `homeless_encampment` - Tents/shelters
- `homeless_person` - People on streets
- `abandoned_vehicle` - Derelict vehicles
- `dumped_trash` - Illegally dumped waste
- `manholes` - Utility covers
- `street_signs` - Traffic signs
- `traffic_lights` - Signal lights
- `tyre_marks` - Tire marks
- `graffiti` - Spray paint vandalism

## ⚡ Performance Tips

### Faster Processing
```bash
# Use fewer categories (3-5 instead of all 15)
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes alligator_cracks --fps 2.0

# Lower FPS for videos
python main_sam3_only.py --mode video --input video.mp4 --fps 1.0

# Skip visualization
python main_sam3_only.py --mode video --input video.mp4 --no-viz
```

### Memory Optimization
```bash
# If running out of memory, use fewer categories
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes alligator_cracks --max-frames 20
```

## 📁 Output Files

After running, check the output directory:

```
output_sam3_only/
├── image_name_sam3_detections.json      # Detection data
├── image_name_sam3_annotated.jpg        # Visualized image
├── video_name_sam3_detections.json      # Video detection data
└── video_name_sam3_annotated.mp4        # Annotated video
```

## 🆚 When to Use SAM3-Only vs Full System

### Use SAM3-Only (`main_sam3_only.py`) When:
- ✅ Need quick results
- ✅ Limited GPU memory (<12 GB)
- ✅ Testing/prototyping
- ✅ Know specific categories to detect

### Use Full Agentic System (`main_simple.py`) When:
- ✅ Need highest accuracy
- ✅ Production deployment
- ✅ Have sufficient GPU memory (12+ GB)
- ✅ Complex scenes with many object types

## 🔧 Troubleshooting

### "Out of memory" Error
```bash
# Use fewer categories or frames
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes --max-frames 10
```

### "Model not found" Error
```bash
# Login and accept terms
huggingface-cli login
# Then visit: https://huggingface.co/facebook/sam3
```

### Slow Processing
```bash
# Check you're using GPU
python main_sam3_only.py --mode image --input image.jpg --device cuda

# Or reduce categories
python main_sam3_only.py --mode video --input video.mp4 \
    --categories potholes alligator_cracks
```

## 📖 Full Documentation

See `README_SAM3_ONLY.md` for detailed documentation.

## 🎯 Recommended Workflow

1. **Test with single image first:**
   ```bash
   python main_sam3_only.py --mode image --input test_image.jpg \
       --categories potholes
   ```

2. **Test with few frames:**
   ```bash
   python main_sam3_only.py --mode video --input test_video.mp4 \
       --max-frames 5 --fps 2.0
   ```

3. **Run full video:**
   ```bash
   python main_sam3_only.py --mode video --input full_video.mp4 \
       --categories potholes alligator_cracks damaged_crosswalks \
       --fps 2.0
   ```
