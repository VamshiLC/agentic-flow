# How to Exclude Hallucinated Categories

## Problem

The model sometimes hallucinates detections for certain categories (especially **graffiti** and **tyre_marks**) even when they're not present in the image.

## Quick Fix: Completely Exclude Categories

### Option 1: Exclude in Code

Edit your detection script to exclude problematic categories:

```python
from detector_unified import get_detector

# Exclude graffiti completely (recommended if it's hallucinating)
detector = get_detector(
    categories=None,  # Detect all categories
    exclude_categories=["graffiti", "tyre_marks"]  # But exclude these
)

result = detector.detect_infrastructure(image)
```

### Option 2: Exclude via Command Line

Update your main scripts to support `--exclude` parameter.

**Edit `main_simple.py`** - Add after other arguments:

```python
parser.add_argument(
    "--exclude",
    nargs="+",
    help="Categories to exclude (e.g., --exclude graffiti tyre_marks)"
)
```

Then in the detector initialization:

```python
detector = UnifiedInfrastructureDetector(
    model_name=args.model,
    categories=args.categories,
    device=args.device,
    use_quantization=args.quantize,
    low_memory=args.low_memory,
    exclude_categories=args.exclude  # Add this line
)
```

**Usage:**
```bash
# Exclude graffiti
python main_simple.py --mode video --input video.mp4 --exclude graffiti

# Exclude multiple categories
python main_simple.py --mode video --input video.mp4 --exclude graffiti tyre_marks
```

## Automatic Filtering (Already Active)

The filter system is **already active** and reduces graffiti false positives by:

1. **Very high confidence threshold**: 0.90 (most hallucinations are lower)
2. **Position check**: Graffiti not in top 20% of image (sky area)
3. **Aspect ratio check**: Filters extremely thin/wide detections
4. **Size constraints**: Must be between 2,000 - 500,000 pixels

### Graffiti Filter Rules

```python
# In detection_filters.py
"graffiti": 0.90,  # Requires 90% confidence (vs typical 0.70)

# Additional checks:
- Not in top 20% of image (sky)
- Aspect ratio between 0.2 - 3.0 (not too thin/wide)
- Reasonable size (2K - 500K pixels)
```

## Comparison: Filter vs Exclude

| Method | Graffiti Detections | When to Use |
|--------|---------------------|-------------|
| **Filter (default)** | Only high-confidence (>0.90) | You sometimes need graffiti detection |
| **Exclude** | Zero detections | Never need graffiti detection |

## Tuning Filter Thresholds

If graffiti is still appearing, increase the threshold in `detection_filters.py`:

```python
self.min_confidence = {
    "graffiti": 0.95,  # Increase from 0.90 to 0.95
}
```

Or make position check stricter:

```python
# In _validate_graffiti method
if center_y < image_height * 0.3:  # Was 0.2 (top 20% -> top 30%)
    return False
```

## Other Categories Prone to Hallucination

Based on testing, these categories also hallucinate frequently:

| Category | Default Threshold | Recommendation |
|----------|-------------------|----------------|
| **graffiti** | 0.90 | Exclude or increase to 0.95 |
| **tyre_marks** | 0.85 | Consider excluding |
| **damaged_paint** | 0.70 | Usually OK |
| **manholes** | 0.70 | Usually OK |

## Testing with Exclusions

```bash
# Test with graffiti excluded
python test_single_image.py \
    --input test_image.jpg \
    --output results/ \
    --exclude graffiti tyre_marks

# Check the output
cat results/test_image_detections.json | grep -E "label|num_detections"
```

## Monitoring Filter Activity

To see what's being filtered, run with logging:

```bash
# Enable debug logging
export PYTHONUNBUFFERED=1

python main_simple.py --mode video --input video.mp4 2>&1 | grep -E "Filtered|graffiti"
```

You'll see output like:
```
Filtered potholes 0.80: aspect_ratio=0.65
Filtered graffiti 0.85: confidence < 0.90
Filtered graffiti 0.88: in top 20% of image
```

## Summary

**For hallucinated graffiti:**

1. ✅ **Best**: Completely exclude if you never need it
   ```python
   exclude_categories=["graffiti"]
   ```

2. ✅ **Good**: Use automatic filtering (already active, 0.90 threshold)

3. ⚙️ **Fine-tune**: Increase threshold to 0.95+ if still seeing false positives

The filtering system will **automatically** reduce most graffiti hallucinations. Only exclude if you're seeing persistent false positives even with the 0.90 threshold.
