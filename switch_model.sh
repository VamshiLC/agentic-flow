#!/bin/bash
# Quick script to switch between models

MODEL=$1

if [ "$MODEL" == "4b" ]; then
    echo "Switching to Qwen3-VL-4B..."
    sed -i 's/Qwen2.5-VL-7B/Qwen3-VL-4B/g' config.py detector_unified.py agent/detection_agent_hf.py main_simple.py
    echo "✓ Switched to Qwen3-VL-4B"
elif [ "$MODEL" == "7b" ]; then
    echo "Switching to Qwen2.5-VL-7B..."
    sed -i 's/Qwen3-VL-4B/Qwen2.5-VL-7B/g' config.py detector_unified.py agent/detection_agent_hf.py main_simple.py
    echo "✓ Switched to Qwen2.5-VL-7B"
else
    echo "Usage: bash switch_model.sh [4b|7b]"
    echo "  4b = Qwen3-VL-4B-Instruct (known working)"
    echo "  7b = Qwen2.5-VL-7B-Instruct (testing)"
fi
