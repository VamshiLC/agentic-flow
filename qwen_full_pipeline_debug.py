#!/usr/bin/env python3
"""
Qwen2.5-VL-7B Full Pipeline: Face Blur + Damaged Crosswalk Detection
ENHANCED DEBUG VERSION - Shows face bounding boxes and detailed logging
"""
import sys
import re
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from models.qwen_direct_loader import Qwen3VLDirectDetector


# Prompts
FACE_DETECTION_PROMPT = """Analyze this image and detect all human faces.

For each face you detect, provide the output in this exact format:
Face: <number>, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

If NO faces are detected, respond with: "No faces detected"

Now analyze the image:"""


INFRASTRUCTURE_PROMPT = """Analyze this road infrastructure image and detect damaged crosswalks.

What to detect:
- Damaged Crosswalk: Pedestrian crosswalk markings that are faded, worn, cracked, or damaged

Look for:
- Faded white stripes/zebra crossings
- Worn pedestrian crossing markings
- Damaged or incomplete crosswalk paint
- Deteriorated crosswalk lines

For each damaged crosswalk detected, output in this exact format:
Defect: Damaged Crosswalk, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

If NO damaged crosswalks detected, respond with: "No defects detected"

Now analyze the image:"""


def parse_face_detections(response, image_size, confidence_threshold=0.5):
    """Parse face detections from Qwen response."""
    bboxes = []
    width, height = image_size

    print(f"\n  [DEBUG] Parsing face detections from response...")
    print(f"  [DEBUG] Image size: {width}x{height}")
    print(f"  [DEBUG] Response text: {response[:500]}...")

    if "no faces detected" in response.lower():
        print(f"  [DEBUG] Response indicates no faces detected")
        return bboxes

    pattern = r'Face:\s*\d+,\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)'
    matches = re.findall(pattern, response, re.IGNORECASE)

    print(f"  [DEBUG] Regex found {len(matches)} matches")

    for i, match in enumerate(matches):
        try:
            x1, y1, x2, y2, conf = map(float, match)
            print(f"  [DEBUG] Match {i+1}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}")

            if conf < confidence_threshold:
                print(f"  [DEBUG]   -> Skipped (confidence {conf} < {confidence_threshold})")
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clamp to image bounds
            x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if (x1_orig, y1_orig, x2_orig, y2_orig) != (x1, y1, x2, y2):
                print(f"  [DEBUG]   -> Clamped from [{x1_orig},{y1_orig},{x2_orig},{y2_orig}] to [{x1},{y1},{x2},{y2}]")

            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))
                print(f"  [DEBUG]   -> Added bbox: [{x1},{y1},{x2},{y2}] (size: {x2-x1}x{y2-y1})")
            else:
                print(f"  [DEBUG]   -> Invalid bbox (x2={x2} <= x1={x1} or y2={y2} <= y1={y1})")

        except (ValueError, IndexError) as e:
            print(f"  [DEBUG] Match {i+1} parsing failed: {e}")
            continue

    print(f"  [DEBUG] Final: {len(bboxes)} valid face bboxes\n")
    return bboxes


def parse_infrastructure_detections(response, image_size, confidence_threshold=0.5):
    """Parse infrastructure detections from Qwen response."""
    detections = []
    width, height = image_size

    if "no defects detected" in response.lower():
        return detections

    pattern = r'Defect:\s*([^,\n]+),\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)'
    matches = re.findall(pattern, response, re.IGNORECASE)

    for match in matches:
        try:
            category, x1, y1, x2, y2, conf = match
            category = category.strip()
            x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(conf)

            if conf < confidence_threshold:
                continue

            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 > x1 and y2 > y1:
                detections.append({
                    'category': category,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        except (ValueError, IndexError):
            continue

    return detections


def blur_faces(image, bboxes, strength=51, padding=20):
    """Apply Gaussian blur to face regions."""
    if len(bboxes) == 0:
        print(f"  [DEBUG] No bboxes to blur, returning original image")
        return image.copy()

    output = image.copy()
    h, w = image.shape[:2]

    print(f"\n  [DEBUG] Blurring {len(bboxes)} face region(s)...")
    print(f"  [DEBUG] Image shape: {h}x{w}, Blur strength: {strength}, Padding: {padding}")

    if strength % 2 == 0:
        strength += 1
        print(f"  [DEBUG] Adjusted blur strength to odd number: {strength}")

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        print(f"  [DEBUG] Face {i+1}: [{x1},{y1},{x2},{y2}]")

        # Add padding
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        print(f"  [DEBUG]   -> With padding: [{x1_pad},{y1_pad},{x2_pad},{y2_pad}]")

        face_region = output[y1_pad:y2_pad, x1_pad:x2_pad]
        print(f"  [DEBUG]   -> Region shape: {face_region.shape}")

        if face_region.size == 0:
            print(f"  [DEBUG]   -> Region is empty, skipping")
            continue

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(face_region, (strength, strength), 0)
        output[y1_pad:y2_pad, x1_pad:x2_pad] = blurred
        print(f"  [DEBUG]   -> Blur applied successfully")

    print(f"  [DEBUG] Blur complete!\n")
    return output


def draw_face_boxes(image, bboxes):
    """Draw bounding boxes around detected faces."""
    output = image.copy()

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Draw green box around face
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw label
        label = f"Face {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return output


def draw_detections(image, detections):
    """Draw bounding boxes for infrastructure detections."""
    output = image.copy()

    # Category colors (BGR)
    colors = {
        'damaged crosswalk': (0, 0, 255),  # Red
        'crosswalk': (0, 0, 255),          # Red
    }

    for det in detections:
        category = det['category'].lower()
        bbox = det['bbox']
        conf = det['confidence']

        x1, y1, x2, y2 = bbox
        color = colors.get(category, (128, 128, 128))

        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{category} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python qwen_full_pipeline_debug.py <input_image> [output_dir]")
        print()
        print("Examples:")
        print("  python qwen_full_pipeline_debug.py road.jpg")
        print("  python qwen_full_pipeline_debug.py road.jpg results/")
        print()
        print("This DEBUG version:")
        print("  1. Detects and blurs faces (privacy)")
        print("  2. Draws GREEN boxes around detected faces (NEW!)")
        print("  3. Detects damaged crosswalks")
        print("  4. Shows detailed debug logging")
        print("  5. Saves annotated results")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "qwen_results_debug"

    if not Path(input_path).exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("Qwen2.5-VL-7B Full Pipeline - DEBUG VERSION")
    print("Face Blur + Damaged Crosswalk Detection")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}/\n")

    # Read image
    image_cv = cv2.imread(input_path)
    if image_cv is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    image_pil = Image.open(input_path).convert('RGB')
    h, w = image_cv.shape[:2]
    print(f"Image size: {w}x{h}")

    # Save original
    cv2.imwrite(str(output_path / "0_original.jpg"), image_cv)

    # Load Qwen (once for both tasks)
    print("\n" + "=" * 70)
    print("STEP 1: Loading Qwen2.5-VL-7B Model")
    print("=" * 70)

    try:
        detector = Qwen3VLDirectDetector(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            use_quantization=False,
            low_memory=False
        )
        print("✓ Qwen loaded (will be used for both face detection and infrastructure)\n")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 1: Detect and blur faces
    print("=" * 70)
    print("STEP 2: Face Detection & Privacy Blur")
    print("=" * 70)

    try:
        face_result = detector.detect(image_pil, FACE_DETECTION_PROMPT, max_new_tokens=512)

        if not face_result.get('success', False):
            print("✗ Face detection failed")
            face_bboxes = []
        else:
            face_response = face_result.get('text', '')
            print(f"Qwen raw response:\n{'-'*70}")
            print(face_response)
            print(f"{'-'*70}")

            face_bboxes = parse_face_detections(face_response, (w, h))

        print(f"\n✓ Detected {len(face_bboxes)} face(s)")

        if face_bboxes:
            print("\nFace bounding boxes:")
            for i, (x1, y1, x2, y2) in enumerate(face_bboxes):
                print(f"  Face {i+1}: [{x1}, {y1}, {x2}, {y2}] ({x2-x1}x{y2-y1}px)")

            # Draw boxes around faces (NEW!)
            print("\n  Drawing green boxes around faces...")
            face_boxes_image = draw_face_boxes(image_cv, face_bboxes)
            cv2.imwrite(str(output_path / "1a_faces_detected.jpg"), face_boxes_image)
            print("  ✓ Saved: 1a_faces_detected.jpg")

            # Apply blur
            print("\n  Applying Gaussian blur...")
            blurred_image = blur_faces(image_cv, face_bboxes, strength=51)
            cv2.imwrite(str(output_path / "1b_privacy_protected.jpg"), blurred_image)
            print("  ✓ Saved: 1b_privacy_protected.jpg")
            print("  ✓ Privacy blur applied")
        else:
            print("\n  No faces detected, using original image")
            blurred_image = image_cv.copy()
            cv2.imwrite(str(output_path / "1b_privacy_protected.jpg"), blurred_image)

    except Exception as e:
        print(f"✗ ERROR during face detection: {e}")
        import traceback
        traceback.print_exc()
        blurred_image = image_cv.copy()
        face_bboxes = []

    # Step 2: Detect damaged crosswalks (on blurred image)
    print("\n" + "=" * 70)
    print("STEP 3: Damaged Crosswalk Detection")
    print("=" * 70)

    try:
        # Convert blurred image to PIL
        blurred_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))

        infra_result = detector.detect(blurred_pil, INFRASTRUCTURE_PROMPT, max_new_tokens=1024)

        if not infra_result.get('success', False):
            print("✗ Crosswalk detection failed")
            detections = []
        else:
            infra_response = infra_result.get('text', '')
            print(f"Qwen response: {infra_response[:300]}...")
            detections = parse_infrastructure_detections(infra_response, (w, h))

        print(f"\n✓ Detected {len(detections)} damaged crosswalk(s)")

        if detections:
            print(f"\nDamaged crosswalks found: {len(detections)}")

            # Draw annotations
            annotated_image = draw_detections(blurred_image, detections)
            cv2.imwrite(str(output_path / "2_annotated.jpg"), annotated_image)
            print("\n✓ Annotations saved")
        else:
            print("No damaged crosswalks detected")
            annotated_image = blurred_image.copy()
            cv2.imwrite(str(output_path / "2_annotated.jpg"), annotated_image)

    except Exception as e:
        print(f"✗ ERROR during crosswalk detection: {e}")
        import traceback
        traceback.print_exc()
        detections = []

    # Save JSON results
    print("\n" + "=" * 70)
    print("STEP 4: Saving Results")
    print("=" * 70)

    results = {
        "image": str(input_path),
        "image_size": {"width": w, "height": h},
        "faces": {
            "count": len(face_bboxes),
            "boxes": [[int(v) for v in box] for box in face_bboxes]
        },
        "damaged_crosswalks": {
            "count": len(detections),
            "detections": detections
        }
    }

    json_file = output_path / "results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ JSON results: {json_file}")
    print(f"✓ Images saved to: {output_path}/")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Faces detected & blurred:  {len(face_bboxes)}")
    print(f"Damaged crosswalks found:  {len(detections)}")
    print()
    print("Output files:")
    print(f"  0_original.jpg           - Original image")
    print(f"  1a_faces_detected.jpg    - GREEN boxes around detected faces (NEW!)")
    print(f"  1b_privacy_protected.jpg - Faces blurred")
    print(f"  2_annotated.jpg          - Crosswalks annotated")
    print(f"  results.json             - Complete results")
    print()
    print("=" * 70)
    print("DONE! ✅")
    print("=" * 70)

    # Cleanup
    detector.cleanup()


if __name__ == "__main__":
    main()
