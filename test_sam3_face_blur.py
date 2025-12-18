#!/usr/bin/env python3
"""
Test SAM3 Face Blurring on Images

Quick test script to verify SAM3 face detection and blurring works correctly.
"""
import sys
import cv2
import torch
from pathlib import Path
from utils.face_blur import FaceBlurrer
from models.sam3_text_prompt_loader import load_sam3_text_prompt_model


def test_sam3_face_blur_on_image(image_path: str, output_dir: str = "test_sam3_face_output"):
    """Test SAM3 face blurring on a single image."""

    print("="*70)
    print("SAM3 FACE BLURRING TEST")
    print("="*70)
    print(f"Image: {image_path}")
    print()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load SAM3 model
    print("\n1. Loading SAM3 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    try:
        model, processor, loader = load_sam3_text_prompt_model(device=device)
        print("   ✓ SAM3 model loaded")
    except Exception as e:
        print(f"   ✗ ERROR loading SAM3: {e}")
        return

    # Test 1: Gaussian blur
    print("\n2. Testing SAM3 + Gaussian blur...")
    try:
        blurrer_gaussian = FaceBlurrer(
            backend='sam3',
            blur_type='gaussian',
            blur_strength=51,
            sam3_model=model,
            sam3_processor=processor,
            device=device
        )

        print("   Processing...")
        blurred_gaussian, num_faces = blurrer_gaussian.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   ✓ Detected {num_faces} face(s)")

        output_file = output_path / "sam3_gaussian_blur.jpg"
        cv2.imwrite(str(output_file), blurred_gaussian)
        print(f"   ✓ Saved: {output_file}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 2: Pixelation
    print("\n3. Testing SAM3 + Pixelation...")
    try:
        blurrer_pixel = FaceBlurrer(
            backend='sam3',
            blur_type='pixelate',
            blur_strength=20,
            sam3_model=model,
            sam3_processor=processor,
            device=device
        )

        print("   Processing...")
        blurred_pixel, num_faces = blurrer_pixel.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   ✓ Detected {num_faces} face(s)")

        output_file = output_path / "sam3_pixelate.jpg"
        cv2.imwrite(str(output_file), blurred_pixel)
        print(f"   ✓ Saved: {output_file}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 3: Strong blur
    print("\n4. Testing SAM3 + Strong blur...")
    try:
        blurrer_strong = FaceBlurrer(
            backend='sam3',
            blur_type='gaussian',
            blur_strength=101,
            sam3_model=model,
            sam3_processor=processor,
            device=device
        )

        print("   Processing...")
        blurred_strong, num_faces = blurrer_strong.blur_faces(
            image.copy(),
            return_face_count=True
        )
        print(f"   ✓ Detected {num_faces} face(s)")

        output_file = output_path / "sam3_strong_blur.jpg"
        cv2.imwrite(str(output_file), blurred_strong)
        print(f"   ✓ Saved: {output_file}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Save original for comparison
    original_file = output_path / "original.jpg"
    cv2.imwrite(str(original_file), image)

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_path}/")
    print("\nGenerated files:")
    print(f"  - original.jpg            (original image)")
    print(f"  - sam3_gaussian_blur.jpg  (gaussian blur)")
    print(f"  - sam3_pixelate.jpg       (pixelation)")
    print(f"  - sam3_strong_blur.jpg    (strong blur)")
    print()
    print("SAM3 face detection notes:")
    print("  - Uses text prompt 'face' to detect faces")
    print("  - Provides pixel-precise masks (not just boxes)")
    print("  - Slower than RetinaFace but uses same model as infrastructure")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_sam3_face_blur.py <image_path>")
        print()
        print("Example:")
        print("  python test_sam3_face_blur.py test_image.jpg")
        print()
        print("This will test SAM3-based face blurring and save results to test_sam3_face_output/")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    test_sam3_face_blur_on_image(image_path)


if __name__ == "__main__":
    main()
