#!/usr/bin/env python3
"""
Simple Face Detection Test (No SAM required)

Uses RetinaFace, MediaPipe, or OpenCV - much faster and more accurate than SAM for faces.
"""
import sys
import cv2
import json
from pathlib import Path

# Try to import face detection backends
try:
    from retinaface import RetinaFace as RF
    HAS_RETINAFACE = True
except ImportError:
    HAS_RETINAFACE = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


def draw_face_detections(image, bboxes):
    """Draw bounding boxes around detected faces."""
    output = image.copy()

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Face {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(output, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return output


def detect_faces_retinaface(image):
    """Detect faces using RetinaFace."""
    detections = RF.detect_faces(image)
    bboxes = []

    if detections:
        for key, detection in detections.items():
            facial_area = detection.get('facial_area', None)
            if facial_area:
                x1, y1, x2, y2 = facial_area
                bboxes.append((x1, y1, x2, y2))

    return bboxes


def detect_faces_mediapipe(image):
    """Detect faces using MediaPipe."""
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    bboxes = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            bboxes.append((x1, y1, x2, y2))

    face_detector.close()
    return bboxes


def detect_faces_opencv(image):
    """Detect faces using OpenCV Haar Cascades."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    bboxes = []
    for (x, y, w, h) in faces:
        bboxes.append((x, y, x + w, y + h))

    return bboxes


def blur_faces(image, bboxes, blur_type='gaussian', strength=51):
    """Blur detected faces."""
    output = image.copy()

    for (x1, y1, x2, y2) in bboxes:
        face_region = output[y1:y2, x1:x2]

        if face_region.size == 0:
            continue

        if blur_type == 'gaussian':
            if strength % 2 == 0:
                strength += 1
            blurred = cv2.GaussianBlur(face_region, (strength, strength), 0)
        else:  # pixelate
            h, w = face_region.shape[:2]
            temp = cv2.resize(face_region, (w // strength, h // strength), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        output[y1:y2, x1:x2] = blurred

    return output


def test_face_detection(image_path, output_dir="test_face_detections"):
    """Test face detection with available backends."""

    print("="*70)
    print("FACE DETECTION TEST (Specialized Models)")
    print("="*70)
    print(f"Image: {image_path}\n")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save original
    original_file = output_path / "0_original.jpg"
    cv2.imwrite(str(original_file), image)

    # Try different backends
    backends_tested = []

    # Test RetinaFace
    if HAS_RETINAFACE:
        print("1. Testing RetinaFace (BEST accuracy)...")
        try:
            bboxes = detect_faces_retinaface(image)
            print(f"   ✓ Detected {len(bboxes)} face(s)")

            if bboxes:
                detected = draw_face_detections(image, bboxes)
                cv2.imwrite(str(output_path / "1_retinaface_detections.jpg"), detected)

                blurred = blur_faces(image, bboxes)
                cv2.imwrite(str(output_path / "2_retinaface_blurred.jpg"), blurred)

                backends_tested.append(("RetinaFace", len(bboxes), bboxes))
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("1. RetinaFace: NOT INSTALLED")
        print("   Install with: pip install retinaface")

    print()

    # Test MediaPipe
    if HAS_MEDIAPIPE:
        print("2. Testing MediaPipe (FAST)...")
        try:
            bboxes = detect_faces_mediapipe(image)
            print(f"   ✓ Detected {len(bboxes)} face(s)")

            if bboxes:
                detected = draw_face_detections(image, bboxes)
                cv2.imwrite(str(output_path / "3_mediapipe_detections.jpg"), detected)

                blurred = blur_faces(image, bboxes)
                cv2.imwrite(str(output_path / "4_mediapipe_blurred.jpg"), blurred)

                backends_tested.append(("MediaPipe", len(bboxes), bboxes))
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("2. MediaPipe: NOT INSTALLED")
        print("   Install with: pip install mediapipe")

    print()

    # Test OpenCV (always available)
    print("3. Testing OpenCV Haar Cascades (BASIC)...")
    try:
        bboxes = detect_faces_opencv(image)
        print(f"   ✓ Detected {len(bboxes)} face(s)")

        if bboxes:
            detected = draw_face_detections(image, bboxes)
            cv2.imwrite(str(output_path / "5_opencv_detections.jpg"), detected)

            blurred = blur_faces(image, bboxes)
            cv2.imwrite(str(output_path / "6_opencv_blurred.jpg"), blurred)

            backends_tested.append(("OpenCV", len(bboxes), bboxes))
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Save summary
    if backends_tested:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for backend, count, _ in backends_tested:
            print(f"{backend}: {count} face(s) detected")

        # Save JSON
        summary = {
            "image": str(image_path),
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "backends": [
                {
                    "name": backend,
                    "num_faces": count,
                    "faces": [{"bbox": [int(x1), int(y1), int(x2), int(y2)]}
                             for x1, y1, x2, y2 in bboxes]
                }
                for backend, count, bboxes in backends_tested
            ]
        }

        json_file = output_path / "face_detections_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}/")
        print(f"Detection JSON: {json_file}")
    else:
        print("\nNo backends available! Install at least one:")
        print("  pip install retinaface  # Best")
        print("  pip install mediapipe   # Fast")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_face_detection_simple.py <image_path>")
        print()
        print("This tests face detection with RetinaFace, MediaPipe, and OpenCV")
        print("(Much faster and more accurate than SAM for faces!)")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    test_face_detection(image_path)


if __name__ == "__main__":
    main()
