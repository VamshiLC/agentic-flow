#!/usr/bin/env python3
"""
Simple HuggingFace login script for SageMaker
"""
import sys

try:
    from huggingface_hub import login
    print("✓ huggingface_hub found")
except ImportError:
    print("✗ huggingface_hub not installed for this Python")
    print(f"  Python: {sys.executable}")
    print(f"  Version: {sys.version}")
    print("\nInstalling huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "huggingface_hub"])
    from huggingface_hub import login
    print("✓ huggingface_hub installed successfully")

print("\n" + "="*70)
print("HuggingFace Authentication")
print("="*70)
print("\nYou need a HuggingFace token to access SAM3.")
print("\n1. Go to: https://huggingface.co/settings/tokens")
print("2. Click 'New token'")
print("3. Name: 'sagemaker-sam3'")
print("4. Type: 'Read'")
print("5. Copy the token (starts with hf_...)")
print("\n6. IMPORTANT: Request access to SAM3 first!")
print("   Visit: https://huggingface.co/facebook/sam3")
print("   Click 'Agree and access repository'")
print("\n" + "="*70)
print()

# Get token
token = input("Paste your HuggingFace token here: ").strip()

if not token:
    print("✗ No token provided")
    sys.exit(1)

# Login
try:
    login(token=token, add_to_git_credential=True)
    print("\n✓ Successfully authenticated with HuggingFace!")
    print("\nYou can now run: python sam3_detect.py license1.png")
except Exception as e:
    print(f"\n✗ Login failed: {e}")
    sys.exit(1)
