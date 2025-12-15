#!/usr/bin/env python3
"""
Quick test script - NO FILTERS
See raw detections from Qwen2.5-VL-7B
"""
import sys
from pathlib import Path
from PIL import Image
from detector_unified import UnifiedInfrastructureDetector
import json

# Temporarily disable filters by commenting them out
import agent.detection_agent_hf as agent_module

# Monkey patch to skip filtering
original_detect = agent_module.InfrastructureDetectionAgentHF.detect_infrastructure

def detect_no_filter(self, image, filter_categories=None, use_sam3=True):
    """Detect without filters"""
    result = original_detect(self, image, filter_categories, use_sam3)
    print(f"\n🔍 RAW DETECTIONS (before filters): {result.get('raw_detections', 'N/A')}")
    print(f"✅ FINAL DETECTIONS (after filters): {result['num_detections']}")
    if result.get('raw_detections', 0) > result['num_detections']:
        print(f"⚠️  {result.get('raw_detections', 0) - result['num_detections']} detections were filtered out!")
    return result

# Apply monkey patch
agent_module.InfrastructureDetectionAgentHF.detect_infrastructure = detect_no_filter

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_no_filters.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Testing with: {image_path}")
    print("=" * 70)

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Initialize detector
    print("Initializing detector (Qwen2.5-VL-7B)...")
    detector = UnifiedInfrastructureDetector()

    # Detect
    print("\nRunning detection...")
    result = detector.detect_infrastructure(image, use_sam3=True)

    # Show results
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Total detections: {result['num_detections']}")
    print(f"Raw detections: {result.get('raw_detections', 'N/A')}")

    if result['num_detections'] > 0:
        print("\nDetections found:")
        for i, det in enumerate(result['detections'], 1):
            label = det.get('label', 'unknown')
            conf = det.get('confidence', 0.0)
            bbox = det.get('bbox', [])
            print(f"  [{i}] {label}: {conf:.2f} @ {bbox}")
    else:
        print("\n⚠️  No detections found!")
        print("This could mean:")
        print("  1. Qwen2.5-VL-7B is not detecting anything")
        print("  2. Confidence thresholds are too high")
        print("  3. Image quality issue")

    # Save results
    output_json = Path("test_no_filters_result.json")
    with open(output_json, 'w') as f:
        # Make JSON serializable
        json_result = {
            'num_detections': result['num_detections'],
            'raw_detections': result.get('raw_detections', 0),
            'detections': [
                {
                    'label': d.get('label'),
                    'confidence': float(d.get('confidence', 0)),
                    'bbox': d.get('bbox'),
                    'has_mask': d.get('has_mask', False)
                }
                for d in result['detections']
            ]
        }
        json.dump(json_result, f, indent=2)

    print(f"\nResults saved to: {output_json}")
