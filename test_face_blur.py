#!/usr/bin/env python3
"""
Test Face Blurring Functionality

Quick test script to verify face detection and blurring works correctly.
"""
import cv2
import sys
from pathlib import Path
from utils.face_blur import FaceBlurrer, blur_faces_in_frame


def test_face_blur_on_image(image_path: str, output_dir: str = "test_output"):
    """Test face blurring on a single image."""
    print(f"Testing face blurring on: {image_path}")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Test 1: RetinaFace with Gaussian blur
    print("\n1. Testing RetinaFace + Gaussian blur...")
    try:
        blurrer_rf_gaussian = FaceBlurrer(
            backend='retinaface',
            blur_type='gaussian',
            blur_strength=51
        )
        blurred_rf_gaussian, num_faces = blurrer_rf_gaussian.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   Detected {num_faces} face(s)")

        output_file = output_path / "test_retinaface_gaussian.jpg"
        cv2.imwrite(str(output_file), blurred_rf_gaussian)
        print(f"   Saved: {output_file}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 2: MediaPipe with Gaussian blur
    print("\n2. Testing MediaPipe + Gaussian blur...")
    try:
        blurrer_mp_gaussian = FaceBlurrer(
            backend='mediapipe',
            blur_type='gaussian',
            blur_strength=51
        )
        blurred_mp_gaussian, num_faces = blurrer_mp_gaussian.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   Detected {num_faces} face(s)")

        output_file = output_path / "test_mediapipe_gaussian.jpg"
        cv2.imwrite(str(output_file), blurred_mp_gaussian)
        print(f"   Saved: {output_file}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 3: MediaPipe with Pixelation
    print("\n3. Testing MediaPipe + Pixelation...")
    try:
        blurrer_mp_pixel = FaceBlurrer(
            backend='mediapipe',
            blur_type='pixelate',
            blur_strength=20  # Smaller = larger pixels
        )
        blurred_mp_pixel, num_faces = blurrer_mp_pixel.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   Detected {num_faces} face(s)")

        output_file = output_path / "test_mediapipe_pixelate.jpg"
        cv2.imwrite(str(output_file), blurred_mp_pixel)
        print(f"   Saved: {output_file}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 4: OpenCV Haar Cascade with Gaussian blur
    print("\n4. Testing OpenCV Haar Cascade + Gaussian blur...")
    try:
        blurrer_cv_gaussian = FaceBlurrer(
            backend='opencv',
            blur_type='gaussian',
            blur_strength=51
        )
        blurred_cv_gaussian, num_faces = blurrer_cv_gaussian.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   Detected {num_faces} face(s)")

        output_file = output_path / "test_opencv_gaussian.jpg"
        cv2.imwrite(str(output_file), blurred_cv_gaussian)
        print(f"   Saved: {output_file}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test 5: Convenience function (auto-backend)
    print("\n5. Testing convenience function (auto backend)...")
    try:
        blurred_auto = blur_faces_in_frame(image.copy())
        output_file = output_path / "test_auto_backend.jpg"
        cv2.imwrite(str(output_file), blurred_auto)
        print(f"   Saved: {output_file}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print(f"\nAll outputs saved to: {output_path}/")
    print("\nCompare the different methods to choose the best one for your use case:")
    print("  - RetinaFace: Best accuracy, handles challenging angles/lighting (RECOMMENDED)")
    print("  - MediaPipe: Good accuracy, fast, CPU-friendly")
    print("  - OpenCV: Lightweight, fastest, but less accurate (frontal faces only)")
    print("\nBlur types:")
    print("  - Gaussian: Smooth blur effect")
    print("  - Pixelate: Mosaic/censored effect")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_face_blur.py <image_path>")
        print("\nExample:")
        print("  python test_face_blur.py test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    test_face_blur_on_image(image_path)


if __name__ == "__main__":
    main()
