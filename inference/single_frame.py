"""
Single Frame Processing

Process individual frames through the infrastructure detection pipeline.
"""
import os
import json
from agent.detection_agent_hf import InfrastructureDetectionAgentHF
from models.sam3_loader import load_sam3_model
from utils.output_formatter import format_detection_output, save_detection_json


def process_single_frame(
    image_path,
    output_dir="output",
    sam3_processor=None,
    detector=None,
    save_json=True,
    debug=False
):
    """
    Process a single frame through the agent pipeline.

    Args:
        image_path: Path to the image file to process
        output_dir: Directory to save output images and JSON
        sam3_processor: Optional pre-loaded SAM3 processor (if None, will load)
        detector: Optional pre-loaded detector (if None, will create)
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

    # Create detector if not provided
    if detector is None:
        print("Initializing HF-based detection agent...")
        detector = InfrastructureDetectionAgentHF(
            sam3_processor=sam3_processor,
            sam3_confidence=0.3
        )

    # Load image
    from PIL import Image
    image = Image.open(image_path).convert('RGB')

    # Process frame through agent
    print("Running infrastructure detection...")
    result = detector.detect_infrastructure(image, use_sam3=True)

    # Format output
    print("Formatting detection output...")
    num_detections = result.get('num_detections', 0)

    json_output = {
        "frame_id": os.path.basename(image_path),
        "image_path": image_path,
        "num_detections": num_detections,
        "detections": result.get('detections', []),
        "has_masks": result.get('has_masks', False)
    }

    # Print summary
    print(f"\n✓ Detection complete:")
    print(f"  - Detections found: {num_detections}")

    if num_detections > 0:
        print("\n  Detected issues:")
        for i, detection in enumerate(result['detections'], 1):
            category = detection.get("category", "unknown")
            confidence = detection.get("confidence", 0.0)
            print(f"    {i}. {category} (confidence: {confidence:.2f})")

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
