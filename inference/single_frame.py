"""
Single Frame Processing

Process individual frames through the infrastructure detection pipeline.
"""
import os
import json
from agent.detection_agent import InfrastructureDetectionAgent
from models.sam3_loader import load_sam3_model
from models.qwen_loader import get_qwen_config
from utils.output_formatter import format_detection_output, save_detection_json


def process_single_frame(
    image_path,
    output_dir="output",
    sam3_processor=None,
    llm_config=None,
    save_json=True,
    debug=False
):
    """
    Process a single frame through the agent pipeline.

    Args:
        image_path: Path to the image file to process
        output_dir: Directory to save output images and JSON
        sam3_processor: Optional pre-loaded SAM3 processor (if None, will load)
        llm_config: Optional LLM config (if None, will use default)
        save_json: If True, save detection JSON to file
        debug: If True, print debug information

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

    # Load models if not provided
    if sam3_processor is None:
        print("Loading SAM3 model...")
        sam3_processor = load_sam3_model(confidence_threshold=0.5)

    if llm_config is None:
        llm_config = get_qwen_config()

    # Create agent
    print("Initializing detection agent...")
    agent = InfrastructureDetectionAgent(sam3_processor, llm_config)

    # Process frame through agent
    print("Running infrastructure detection...")
    result_image = agent.process_frame(
        image_path,
        output_dir=output_dir,
        debug=debug
    )

    # Format output to JSON
    print("Formatting detection output...")
    json_output = format_detection_output(
        result_image,
        image_path,
        output_dir=output_dir
    )

    # Print summary
    num_detections = len(json_output.get("detections", []))
    print(f"\n✓ Detection complete:")
    print(f"  - Detections found: {num_detections}")
    print(f"  - Output image: {result_image}")

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
