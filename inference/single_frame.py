"""
Single Frame Processing

Process individual frames through the infrastructure detection pipeline.
Uses the NEW HuggingFace-based agent with improved SAM3 segmentation.
"""
import os
import json
from PIL import Image

# Use the NEW HuggingFace agent (not vLLM-based)
from agent.detection_agent_hf import InfrastructureDetectionAgentHF
from models.sam3_loader import load_sam3_model
from utils.output_formatter import format_detection_output, save_detection_json


def process_single_frame(
    image_path,
    output_dir="output",
    sam3_processor=None,
    llm_config=None,  # Kept for backwards compatibility, but not used
    save_json=True,
    debug=False,
    categories=None,
    use_quantization=True
):
    """
    Process a single frame through the agent pipeline.

    Uses the NEW HuggingFace-based agent with:
    - Qwen2.5-VL for intelligent object detection with bounding boxes
    - SAM3 text prompts for high-quality segmentation
    - No vLLM server required

    Args:
        image_path: Path to the image file to process
        output_dir: Directory to save output images and JSON
        sam3_processor: Optional pre-loaded SAM3 processor (if None, will load)
        llm_config: DEPRECATED - kept for backwards compatibility
        save_json: If True, save detection JSON to file
        debug: If True, print debug information
        categories: Optional list of categories to detect
        use_quantization: Use 8-bit quantization for Qwen (default: True)

    Returns:
        dict: Detection output in web app compatible format

    Example:
        result = process_single_frame("frame_001.jpg", output_dir="results/")
    """
    print(f"\n{'='*60}")
    print(f"Processing frame: {image_path}")
    print(f"{'='*60}")

    # Validate input
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load SAM3 if not provided
    if sam3_processor is None:
        print("Loading SAM3 model...")
        sam3_processor = load_sam3_model(confidence_threshold=0.25)

    # Create the NEW HuggingFace-based agent
    print("Initializing detection agent (HuggingFace mode)...")
    agent = InfrastructureDetectionAgentHF(
        sam3_processor=sam3_processor,
        categories=categories,
        use_quantization=use_quantization,
        debug=debug
    )

    # Process frame through agent
    print("Running infrastructure detection...")
    result = agent.detect_infrastructure(
        image_path,
        use_sam3=True
    )

    # Save result image
    result_image_path = None
    if result.get('final_image'):
        result_image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_detected.png")
        result['final_image'].save(result_image_path)
        print(f"  - Output image: {result_image_path}")

    # Format output to JSON (compatible with web app)
    json_output = {
        "frame_id": os.path.basename(image_path),
        "image_path": image_path,
        "output_image": result_image_path,
        "num_detections": result.get('num_detections', 0),
        "detections": []
    }

    # Convert detections to web app format
    for det in result.get('detections', []):
        json_output["detections"].append({
            "category": det.get('category', det.get('label', 'unknown')),
            "confidence": det.get('confidence', 0.0),
            "severity": det.get('severity', 'low'),
            "severityLabel": det.get('severity', 'low').capitalize(),
            "bbox": det.get('bbox', [0, 0, 0, 0]),
            "has_mask": det.get('has_mask', False),
            "description": det.get('description', '')
        })

    # Print summary
    num_detections = json_output["num_detections"]
    print(f"\n✓ Detection complete:")
    print(f"  - Detections found: {num_detections}")

    if num_detections > 0:
        print("\n  Detected issues:")
        for i, detection in enumerate(json_output["detections"], 1):
            category = detection.get("category", "unknown")
            severity = detection.get("severityLabel", "")
            print(f"    {i}. {category} ({severity})")

    # Save JSON if requested
    if save_json:
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(output_dir, f"{frame_name}.json")
        save_detection_json(json_output, json_path)
        print(f"  - JSON saved: {json_path}")

    return json_output


def process_single_frame_simple(image_path, output_dir="output"):
    """
    Simplified single frame processing with default settings.

    Args:
        image_path: Path to the image file
        output_dir: Output directory

    Returns:
        dict: Detection output
    """
    return process_single_frame(
        image_path,
        output_dir=output_dir,
        save_json=True,
        debug=False
    )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python single_frame.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    result = process_single_frame(image_path, output_dir, debug=True)
    print(f"\n{json.dumps(result, indent=2)}")
