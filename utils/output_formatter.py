"""
Output Formatting Utilities

Converts SAM3 agent output to web app compatible JSON schema.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any


# Category mappings to match web app schema
CATEGORY_MAPPINGS = {
    "potholes": {
        "severity": "critical_high",
        "defect_level": "critical_high",
        "severity_color": "red",
        "severity_label": "Severe",
        "type_label": "Pothole"
    },
    "alligator_cracks": {
        "severity": "critical_high",
        "defect_level": "critical_high",
        "severity_color": "red",
        "severity_label": "Severe",
        "type_label": "Alligator Crack"
    },
    "abandoned_vehicles": {
        "severity": "medium",
        "defect_level": "medium",
        "severity_color": "yellow",
        "severity_label": "Moderate",
        "type_label": "Abandoned Vehicle"
    },
    "longitudinal_cracks": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Longitudinal Crack"
    },
    "transverse_cracks": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Transverse Crack"
    },
    "damaged_paint": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Damaged Paint"
    },
    "manholes": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Manhole"
    },
    "dumped_trash": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Dumped Trash"
    },
    "street_signs": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Street Sign"
    },
    "traffic_lights": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Traffic Light"
    },
    "tyre_marks": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Tyre Mark"
    },
    "damaged_crosswalks": {
        "severity": "non_critical_low",
        "defect_level": "non_critical_low",
        "severity_color": "green",
        "severity_label": "Minor",
        "type_label": "Damaged Crosswalk"
    }
}


def parse_sam3_agent_result(result_data):
    """
    Parse SAM3 agent result data.

    Args:
        result_data: Raw result data from SAM3 agent

    Returns:
        list: List of parsed detections with masks, bboxes, confidences
    """
    # TODO: This will need to be implemented based on actual SAM3 agent output format
    # For now, returning a placeholder structure
    detections = []

    # SAM3 agent output parsing will go here
    # The actual implementation depends on the format returned by sam3.agent.inference.run_single_image_inference

    return detections


def format_detection_output(
    sam3_result,
    frame_path,
    llm_response=None,
    output_dir="output"
):
    """
    Convert SAM3 agent output to web app compatible JSON schema.

    Args:
        sam3_result: Output from SAM3 agent (image path or result object)
        frame_path: Path to the input frame
        llm_response: Optional LLM response with detection metadata
        output_dir: Directory where masks are saved

    Returns:
        dict: Formatted detection output matching web app schema

    Output Schema:
    {
        "frame_id": "frame_000001.jpg",
        "detections": [
            {
                "id": "uuid",
                "category": "pothole",
                "bbox": [x1, y1, x2, y2],
                "mask": "path/to/mask.png",
                "confidence": 0.92,
                "severity": "critical_high",
                "defectLevel": "critical_high",
                "severityColor": "red",
                "severityLabel": "Severe",
                "typeLabel": "Pothole",
                "description": "Large pothole at bottom-left"
            }
        ],
        "metadata": {
            "timestamp": "2025-01-15T10:30:00Z",
            "model": "qwen3-vl-4b + sam3",
            "frame_path": "/path/to/frame.jpg"
        }
    }
    """
    output = {
        "frame_id": os.path.basename(frame_path),
        "detections": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "qwen3-vl-4b + sam3",
            "frame_path": os.path.abspath(frame_path)
        }
    }

    # Parse LLM response if available
    if llm_response:
        try:
            if isinstance(llm_response, str):
                llm_data = json.loads(llm_response)
            else:
                llm_data = llm_response

            # Extract detections from LLM response
            if "detections" in llm_data:
                for detection in llm_data["detections"]:
                    formatted_detection = format_single_detection(
                        detection,
                        frame_path,
                        output_dir
                    )
                    if formatted_detection:
                        output["detections"].append(formatted_detection)

        except Exception as e:
            print(f"Warning: Could not parse LLM response: {e}")

    # Parse SAM3 result if available
    if sam3_result:
        # TODO: Parse SAM3 masks/bboxes and merge with LLM detections
        pass

    return output


def format_single_detection(
    detection_data,
    frame_path,
    output_dir="output"
):
    """
    Format a single detection to match web app schema.

    Args:
        detection_data: Detection data from LLM
        frame_path: Path to the input frame
        output_dir: Directory where masks are saved

    Returns:
        dict: Formatted detection
    """
    category = detection_data.get("category", "").lower()

    # Normalize category name (handle variations)
    if category not in CATEGORY_MAPPINGS:
        # Try to find closest match
        print(f"Warning: Unknown category '{category}', skipping")
        return None

    # Get category mapping
    category_info = CATEGORY_MAPPINGS[category]

    # Create formatted detection
    formatted = {
        "id": str(uuid.uuid4()),
        "category": category,
        "typeLabel": category_info["type_label"],
        "severity": category_info["severity"],
        "defectLevel": category_info["defect_level"],
        "severityColor": category_info["severity_color"],
        "severityLabel": category_info["severity_label"],
        "description": detection_data.get("description", ""),
        "confidence": detection_data.get("confidence", 0.0),
        "bbox": detection_data.get("bbox", []),
        "mask": detection_data.get("mask", ""),
    }

    return formatted


def save_detection_json(detection_output, output_path):
    """
    Save detection output to JSON file.

    Args:
        detection_output: Formatted detection output
        output_path: Path to save JSON file

    Returns:
        str: Path to saved JSON file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(detection_output, f, indent=2)

    return output_path


def create_detection_summary(detection_outputs):
    """
    Create a summary of detections across multiple frames.

    Args:
        detection_outputs: List of detection outputs from multiple frames

    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_frames": len(detection_outputs),
        "total_detections": 0,
        "detections_by_category": {},
        "detections_by_severity": {
            "critical_high": 0,
            "medium": 0,
            "non_critical_low": 0
        },
        "frames_with_issues": 0
    }

    for frame_output in detection_outputs:
        detections = frame_output.get("detections", [])

        if detections:
            summary["frames_with_issues"] += 1

        for detection in detections:
            summary["total_detections"] += 1

            # Count by category
            category = detection.get("category", "unknown")
            summary["detections_by_category"][category] = \
                summary["detections_by_category"].get(category, 0) + 1

            # Count by severity
            severity = detection.get("severity", "non_critical_low")
            summary["detections_by_severity"][severity] = \
                summary["detections_by_severity"].get(severity, 0) + 1

    return summary
