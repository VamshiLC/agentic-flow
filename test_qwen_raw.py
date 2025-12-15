#!/usr/bin/env python3
"""
Test Qwen2.5-VL-7B raw response to see if it's detecting anything
"""
import torch
from PIL import Image
from models.qwen_direct_loader import Qwen3VLDirectDetector

# Test with a simple image
print("="*70)
print("Testing Qwen2.5-VL-7B Raw Response")
print("="*70)

# Initialize detector
print("\nLoading Qwen2.5-VL-7B...")
detector = Qwen3VLDirectDetector(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device="cuda",
    use_quantization=False,
    low_memory=False
)

# Create a test image path - use one from the video
test_image_path = input("\nEnter path to test image (e.g., /path/to/frame.jpg): ").strip()

try:
    image = Image.open(test_image_path).convert('RGB')
    print(f"✓ Loaded image: {image.size}")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    exit(1)

# Test detection with simple prompt
print("\n" + "="*70)
print("TEST 1: Simple Detection Prompt")
print("="*70)

simple_prompt = """Analyze this road/street image and identify ANY visible infrastructure issues or objects.

Look for:
- Potholes (holes in road)
- Cracks in pavement
- Homeless people or encampments
- Abandoned vehicles
- Manholes
- Damaged road markings
- Graffiti
- Any other infrastructure problems

For EACH item you see, provide:
Defect: <name>, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

Use normalized coordinates (0-1000 scale).

If you see NOTHING, respond with: "No defects detected"
"""

response = detector.generate_response(image, simple_prompt)

print("\nMODEL RESPONSE:")
print("-"*70)
print(response)
print("-"*70)

# Check if model is responding correctly
if "no defects" in response.lower() or len(response.strip()) < 10:
    print("\n⚠️  MODEL DID NOT DETECT ANYTHING")
    print("This could mean:")
    print("  1. The image truly has no issues")
    print("  2. Qwen2.5-VL-7B needs different prompt format")
    print("  3. Model confidence thresholds are too high")
else:
    print("\n✓ Model is responding with detections!")

# Test with more explicit prompt
print("\n" + "="*70)
print("TEST 2: Very Explicit Prompt")
print("="*70)

explicit_prompt = """You are an infrastructure inspector analyzing a road image.

YOUR TASK: Find and report EVERY visible problem or object in this image.

WHAT TO LOOK FOR (mark ALL that you see):
1. Potholes - any holes or depressions in the road
2. Cracks - any visible cracks in pavement (small or large)
3. People - any people visible, especially homeless individuals
4. Vehicles - cars, trucks, especially if abandoned
5. Manholes - circular metal covers
6. Damaged paint - faded road markings
7. Graffiti - any spray paint or tags
8. Trash - piles of garbage or debris
9. ANY other infrastructure issues

RESPONSE FORMAT (for EACH item you find):
Defect: <type>, Box: [x1, y1, x2, y2], Confidence: <score>

Coordinates: Use 0-1000 scale (top-left is 0,0, bottom-right is 1000,1000)

IMPORTANT: Report EVERYTHING you see, even if you're not 100% certain.

If you see ABSOLUTELY NOTHING problematic, write exactly: "No defects detected"
"""

response2 = detector.generate_response(image, explicit_prompt)

print("\nMODEL RESPONSE:")
print("-"*70)
print(response2)
print("-"*70)

if "no defects" in response2.lower():
    print("\n⚠️  STILL NO DETECTIONS")
    print("\nPossible issues:")
    print("  1. Qwen2.5-VL-7B may need fine-tuning for this task")
    print("  2. Try switching back to Qwen3-VL-4B")
    print("  3. Or use Qwen2-VL-7B instead")
else:
    print("\n✓ Got detections with explicit prompt!")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
