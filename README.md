# ASH Infrastructure Detection Agent

An autonomous AI agent for detecting and segmenting road infrastructure issues from GoPro footage using **Qwen3-VL-4B-Instruct** and **SAM3** (Segment Anything Model 3).

## Overview

This agent autonomously analyzes road images to detect 12 categories of infrastructure issues:

### Critical Issues (Red)
- **Potholes** - Severe road defects
- **Alligator Cracks** - Web-like cracking patterns

### Medium Priority (Yellow)
- **Abandoned Vehicles** - Derelict vehicles on/near roads

### Low Priority (Green)
- **Longitudinal Cracks** - Cracks parallel to traffic
- **Transverse Cracks** - Cracks perpendicular to traffic
- **Damaged Paint** - Deteriorated road markings
- **Manholes** - Utility access points
- **Dumped Trash** - Debris and litter
- **Street Signs** - Traffic/regulatory signs
- **Traffic Lights** - Signal lights and poles
- **Tyre Marks** - Tire/skid marks
- **Damaged Crosswalks** - Faded crosswalk markings

## Architecture

```
GoPro Frame → Qwen3-VL Agent (analyzes) → SAM3 Tool (segments) → JSON Output
```

**Based on Meta's official SAM3 Agent pattern:**
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb

### Key Components

1. **Qwen3-VL-4B-Instruct** (via vLLM) - Vision-language model for autonomous detection
2. **SAM3** - Segmentation tool called by the agent
3. **Agent Loop** - Orchestrates MLLM → SAM3 → Output formatting
4. **Video Processor** - Extracts frames and batch processes

## Project Structure

```
sam3-agent/
├── agent/
│   ├── __init__.py
│   ├── detection_agent.py      # Main agent class
│   ├── prompts.py              # System prompts for detection
│   └── tools.py                # SAM3 tool wrapper (original)
├── inference/
│   ├── __init__.py
│   ├── single_frame.py         # Process one frame
│   └── video_processor.py      # Process full video
├── models/
│   ├── __init__.py
│   ├── sam3_loader.py          # SAM3 model loading
│   └── qwen_loader.py          # Qwen vLLM config
├── utils/
│   ├── __init__.py
│   ├── video_utils.py          # Frame extraction
│   └── output_formatter.py     # JSON formatting
├── config.py                   # Centralized configuration
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── setup_sagemaker.sh          # SageMaker setup script
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- AWS SageMaker or local machine with GPU

### Option 1: Automated Setup (SageMaker)

```bash
chmod +x setup_sagemaker.sh
./setup_sagemaker.sh
```

This will:
1. Install SAM3 from source
2. Install Python dependencies
3. Create vLLM conda environment
4. Start Qwen3-VL-4B server on port 8001

### Option 2: Manual Setup

1. **Install SAM3:**
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup vLLM (separate conda env recommended):**
```bash
conda create -n vllm python=3.12
conda activate vllm
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

4. **Start vLLM server:**
```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --tensor-parallel-size 1 \
    --allowed-local-media-path / \
    --enforce-eager \
    --port 8001
```

## Usage

### Process a Single Image

```bash
python main.py --mode image --input gopro_frame.jpg --output ./results
```

### Process a Video

```bash
# Process 1 frame per second (sample rate 30 for 30fps video)
python main.py --mode video --input gopro_video.mp4 --output ./results --sample-rate 30

# Process with time range
python main.py --mode video --input video.mp4 --start-time 10 --end-time 60 --output ./results

# Process every frame (slower but more comprehensive)
python main.py --mode video --input video.mp4 --sample-rate 1 --output ./results
```

### Other Commands

```bash
# Check if vLLM server is running
python main.py --check-server

# List available Qwen3-VL models
python main.py --list-models

# Enable debug mode
python main.py --mode image --input frame.jpg --output ./results --debug

# Output JSON to console
python main.py --mode image --input frame.jpg --output ./results --json
```

## Output Format

### Single Frame Output

```json
{
  "frame_id": "frame_000001.jpg",
  "detections": [
    {
      "id": "uuid-here",
      "category": "pothole",
      "typeLabel": "Pothole",
      "severity": "critical_high",
      "defectLevel": "critical_high",
      "severityColor": "red",
      "severityLabel": "Severe",
      "description": "Large pothole at bottom-left of frame",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2],
      "mask": "path/to/mask.png"
    }
  ],
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "model": "qwen3-vl-4b + sam3",
    "frame_path": "/path/to/frame.jpg"
  }
}
```

### Video Processing Output

```
output/
├── frames/                    # Extracted frames
│   ├── frame_000001.jpg
│   └── frame_000002.jpg
├── detections/                # Per-frame detection JSONs
│   ├── frame_000001.json
│   └── frame_000002.json
├── video_detections.json      # Consolidated results
└── summary.json               # Statistics summary
```

### Summary Statistics

```json
{
  "total_frames": 100,
  "total_detections": 45,
  "frames_with_issues": 32,
  "detections_by_category": {
    "potholes": 12,
    "alligator_cracks": 8,
    "longitudinal_cracks": 15,
    ...
  },
  "detections_by_severity": {
    "critical_high": 20,
    "medium": 5,
    "non_critical_low": 20
  }
}
```

## Configuration

Edit `config.py` to customize:

```python
# Model settings
QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_SERVER_URL = "http://0.0.0.0:8001/v1"
SAM3_CONFIDENCE_THRESHOLD = 0.5

# Processing settings
VIDEO_SAMPLE_RATE = 30  # 1 fps at 30fps video

# Output settings
OUTPUT_DIR = "output"
DEBUG_MODE = False
```

Environment variables:
```bash
export QWEN_SERVER_URL="http://localhost:8001/v1"
export SAM3_CONFIDENCE="0.6"
export VIDEO_SAMPLE_RATE="15"  # 2 fps at 30fps
export DEBUG="true"
```

## GPU Requirements

**Estimated VRAM usage:**
- Qwen3-VL-4B-Instruct: ~8GB
- SAM3: ~4GB
- **Total: ~12GB**

**Recommended SageMaker instance:**
- `ml.g4dn.xlarge` (16GB GPU) - Good for most use cases
- `ml.g5.xlarge` (24GB GPU) - Better for high throughput

## Performance Tuning

### Adjust Confidence Threshold

```bash
python main.py --mode image --input frame.jpg --confidence 0.7
```

Higher confidence = fewer false positives, may miss subtle defects

### Frame Sampling Rate

For 30fps video:
- `--sample-rate 30` = 1 fps (recommended for balanced coverage)
- `--sample-rate 15` = 2 fps (more detections, 2x slower)
- `--sample-rate 60` = 0.5 fps (faster, may miss issues)

### Batch Processing

Process multiple videos:
```bash
for video in videos/*.mp4; do
    python main.py --mode video --input "$video" --output "results/$(basename $video .mp4)"
done
```

## Troubleshooting

### vLLM server not starting

Check logs:
```bash
tail -f vllm_server.log
```

Common issues:
- Out of GPU memory → Reduce `--tensor-parallel-size` or use smaller model
- Port 8001 in use → Kill existing process: `pkill -f 'vllm serve'`

### SAM3 import errors

Ensure SAM3 is installed correctly:
```bash
python -c "import sam3; print(sam3.__file__)"
```

If missing, reinstall:
```bash
cd /tmp/sam3
pip install -e .
```

### CUDA out of memory

Reduce batch size or use CPU (very slow):
```python
# In config.py
SAM3_DEVICE = "cpu"
```

## Integration with Web App

The JSON output format matches the web app schema in `ashsensors-internal/web/`:

```typescript
interface Detection {
  id: string;
  category: string;
  severity: string;
  confidence: number;
  bbox: number[];
  // ... matches web app types
}
```

To insert detections into PostgreSQL, parse the JSON and use the web app's API.

## Development

### Run tests

```bash
# Test single frame processing
python inference/single_frame.py test_image.jpg output/

# Test video processing
python inference/video_processor.py test_video.mp4 output/
```

### Add new detection categories

1. Update `config.py` CATEGORIES dict
2. Update `agent/prompts.py` SYSTEM_PROMPT
3. Update `utils/output_formatter.py` CATEGORY_MAPPINGS

## References

- **SAM3:** https://github.com/facebookresearch/sam3
- **SAM3 Agent Example:** https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb
- **Qwen3-VL:** https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- **vLLM:** https://docs.vllm.ai/

## License

This project uses:
- SAM3 (Meta, released November 2025)
- Qwen3-VL (Alibaba Cloud, released October 2025)

See respective repositories for license details.

## Support

For issues or questions:
1. Check `vllm_server.log` for server errors
2. Run with `--debug` flag for verbose output
3. Verify GPU availability: `nvidia-smi`
4. Check vLLM server: `python main.py --check-server`
