# Qwen Pipeline - Face Blur + Damaged Crosswalk Detection

## Quick Start

```bash
python qwen_full_pipeline.py road_image.jpg
```

**What it does:**
1. ✅ Detects and blurs faces (privacy protection)
2. ✅ Detects damaged crosswalks (faded/worn pedestrian crossings)
3. ✅ Uses ONE Qwen model for both tasks
4. ✅ Saves annotated results + JSON

**Output:**
```
qwen_results/
  ├── 0_original.jpg           - Original image
  ├── 1_privacy_protected.jpg  - Faces blurred
  ├── 2_annotated.jpg          - Damaged crosswalks marked (red boxes)
  └── results.json             - Detection data
```

---

## Pipeline Flow

```
Input Image
    ↓
Qwen: Detect Faces
    ↓
Gaussian Blur on Faces (Privacy ✅)
    ↓
Qwen: Detect Damaged Crosswalks (on blurred image)
    ↓
Annotate with Red Boxes
    ↓
Output: Privacy-safe + crosswalk detections
```

**One model, two tasks!**

---

## Example Output

**Console:**
```
Qwen2.5-VL-7B Full Pipeline
Face Blur + Damaged Crosswalk Detection
=============================================

STEP 2: Face Detection & Privacy Blur
✓ Detected 2 face(s)

STEP 3: Damaged Crosswalk Detection
✓ Detected 1 damaged crosswalk(s)

SUMMARY
=============================================
Faces detected & blurred:  2
Damaged crosswalks found:  1
```

---

## Files

- `qwen_full_pipeline.py` - Complete pipeline ⭐
- `qwen_face_blur.py` - Face blur only
- `sam3_detect.py` - SAM3 alternative
