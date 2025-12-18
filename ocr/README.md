# License Plate OCR - Documentation

## Overview

This module detects vehicle license plates and extracts text using a two-model architecture optimized for North American plate formats (US, Canada, Mexico).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT IMAGE                                     │
│                         (from images/ folder)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DETECTION                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     SAM3 (Segment Anything Model 3)                    │  │
│  │                                                                        │  │
│  │  • Text Prompt: "license plate"                                       │  │
│  │  • Output: Bounding boxes [x1, y1, x2, y2] for each plate             │  │
│  │  • Confidence score (0.0 - 1.0)                                       │  │
│  │  • Aspect ratio filter: rejects wheels (ratio < 1.3)                  │  │
│  │                                                                        │  │
│  │  Fallback: Qwen3-VL detection if SAM3 unavailable                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: CROP PLATES                                  │
│                                                                              │
│   For each detected plate:                                                   │
│   • Crop region from original image                                          │
│   • Add 10% padding around bounding box                                      │
│   • Output: Cropped plate images for OCR                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: OCR (Text Extraction)                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Qwen3-VL (Vision Language Model)                   │  │
│  │                                                                        │  │
│  │  • Input: Cropped plate image                                         │  │
│  │  • Model: Qwen/Qwen3-VL-4B-Instruct                                   │  │
│  │  • Output:                                                             │  │
│  │      - plate_text: "ABC1234"                                          │  │
│  │      - ocr_confidence: 0.92                                           │  │
│  │      - state: "California"                                            │  │
│  │      - format: "California standard"                                  │  │
│  │                                                                        │  │
│  │  Optimized prompts for North American plate formats                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                          │
│                                                                              │
│   • Annotated images with bounding boxes and plate text                     │
│   • JSON results with all detection data                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Models Used

| Model | Purpose | Parameters | Memory |
|-------|---------|------------|--------|
| **SAM3** | Detection + Tracking | 848M | ~3GB |
| **Qwen3-VL-4B** | OCR (Text Reading) | 4B | ~8GB (4GB with quantization) |

### SAM3 (Segment Anything Model 3)
- **Role**: Detect license plates in images
- **Input**: Full image + text prompt "license plate"
- **Output**: Bounding boxes, confidence scores, segmentation masks
- **Features**:
  - Text-prompted detection
  - Native video tracking (when processing videos)
  - Aspect ratio filtering to reject false positives (wheels, hubcaps)

### Qwen3-VL-4B-Instruct
- **Role**: Read text from cropped plate images (OCR)
- **Input**: Cropped plate image + OCR prompt
- **Output**: Plate text, confidence, state/province identification
- **Features**:
  - Optimized prompts for North American formats
  - State/province detection from plate patterns
  - Handles partial/obscured plates

---

## Input/Output Format

### Input
```
images/
├── image1.png
├── image2.png
├── image3.jpg
└── ...
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### Output Structure
```
ocr_output/
├── image1_plates.jpg          # Annotated image
├── image2_plates.jpg
├── image3_plates.jpg
└── all_results.json           # Combined results
```

---

## Output JSON Format

### Single Image Result
```json
{
  "image": "images/image1.png",
  "num_plates": 2,
  "plates": [
    {
      "bbox": [120, 450, 280, 500],
      "confidence": 0.95,
      "plate_text": "ABC1234",
      "ocr_confidence": 0.92,
      "state": "California",
      "format": "California standard (1 digit + 3 letters + 3 digits)"
    },
    {
      "bbox": [550, 420, 710, 470],
      "confidence": 0.88,
      "plate_text": "XYZ5678",
      "ocr_confidence": 0.85,
      "state": "Texas",
      "format": "US standard with hyphen"
    }
  ]
}
```

### Batch Results (all_results.json)
```json
{
  "summary": {
    "total_plates": 15,
    "readable_plates": 12,
    "avg_confidence": 0.87,
    "total_images": 11,
    "images_with_plates": 9,
    "states": ["California", "Texas", "Ontario"],
    "plate_texts": ["ABC1234", "XYZ5678", "ABCD123", ...]
  },
  "images": [
    {
      "image": "images/image1.png",
      "num_plates": 2,
      "plates": [...]
    },
    {
      "image": "images/image2.png",
      "num_plates": 1,
      "plates": [...]
    }
  ]
}
```

---

## Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `[x1, y1, x2, y2]` | Bounding box coordinates (pixels) |
| `confidence` | `float` | Detection confidence (0.0 - 1.0) |
| `plate_text` | `string` | Extracted plate text (e.g., "ABC1234") |
| `ocr_confidence` | `float` | OCR confidence (0.0 - 1.0) |
| `state` | `string` | Detected state/province (e.g., "California") |
| `format` | `string` | Plate format description |
| `num_plates` | `int` | Number of plates detected in image |

---

## Supported Plate Formats

### United States
| State | Format | Example |
|-------|--------|---------|
| California | 1 digit + 3 letters + 3 digits | `8ABC123` |
| Texas | 3 letters + hyphen + 4 digits | `ABC-1234` |
| New York | 3 letters + hyphen + 4 digits | `ABC-1234` |
| Florida | 3 letters + space + letter + 2 digits | `ABC A12` |
| Standard | 3 letters + 4 digits | `ABC 1234` |
| Vanity | Up to 7 characters | `COOLCAR` |

### Canada
| Province | Format | Example |
|----------|--------|---------|
| Ontario | 4 letters + 3 digits | `ABCD 123` |
| Quebec | 3 digits + 3 letters | `123 ABC` |
| British Columbia | 2 letters + 1 digit + 2 digits + 1 letter | `AB1 23C` |
| Alberta | 3 letters + 4 digits | `ABC-1234` |

### Mexico
| Format | Example |
|--------|---------|
| 3 letters + 2 digits + 2 digits | `ABC-12-34` |

---

## Usage

### Command Line

```bash
# Process all images in folder
python run_ocr.py --input ./images --output ./ocr_output --quantize

# Single image
python run_ocr.py --input car.jpg --output ./ocr_output --quantize

# Without SAM3 (uses Qwen for detection)
python run_ocr.py --input ./images --output ./ocr_output --quantize --no-sam3
```

### Python API

```python
from ocr import LicensePlateOCR
from ocr.processor import process_image, process_batch_images

# Initialize OCR agent
ocr = LicensePlateOCR(use_quantization=True)

# Process single image
result = ocr.detect_and_read("car.jpg")

for plate in result['plates']:
    print(f"Plate: {plate['plate_text']}")
    print(f"State: {plate['state']}")
    print(f"Confidence: {plate['ocr_confidence']}")

# Process batch of images
from glob import glob
images = sorted(glob("images/*.png"))

results = process_batch_images(
    image_paths=images,
    output_dir="./ocr_output",
    ocr_agent=ocr
)
```

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--quantize` | False | Use 8-bit quantization (reduces memory by ~50%) |
| `--no-sam3` | False | Disable SAM3, use Qwen3-VL for detection |
| `--device` | auto | Device to use (`cuda` or `cpu`) |
| `--low-memory` | False | Enable additional memory optimizations |

---

## Performance

| Configuration | GPU Memory | Speed (per image) |
|---------------|------------|-------------------|
| Full precision | ~12GB | ~2s |
| With `--quantize` | ~6GB | ~2.5s |
| CPU only | ~8GB RAM | ~15s |

---

## Error Handling

| Plate Text | Meaning |
|------------|---------|
| `UNREADABLE` | Plate detected but text could not be extracted |
| `[?]` in text | Specific character unclear (e.g., `AB[?]1234`) |

---

## Directory Structure

```
agentic-flow/
├── images/                    # Input images folder
├── ocr_output/                # Output results folder
├── ocr/
│   ├── __init__.py
│   ├── license_plate_agent.py # Main OCR agent (SAM3 + Qwen)
│   ├── sam3_tracker.py        # SAM3 detection/tracking wrapper
│   ├── processor.py           # Image/video processing
│   ├── prompts.py             # Detection & OCR prompts
│   ├── tracker.py             # IoU-based tracking (fallback)
│   ├── utils.py               # Drawing & validation utilities
│   └── README.md              # This documentation
├── models/
│   ├── qwen_direct_loader.py  # Qwen3-VL model loader
│   └── sam3_loader.py         # SAM3 model loader
└── run_ocr.py                 # CLI entry point
```
