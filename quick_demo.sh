#!/bin/bash
# Quick Demo Script
# Run this for a fast visual check of your model

python quick_qualitative_demo.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/qualitative_config.yaml \
    --output-dir qualitative_results/quick_demo \
    --split val \
    --max-sequences 6 \
    --device cuda

echo "Quick demo complete! Check qualitative_results/quick_demo/"
