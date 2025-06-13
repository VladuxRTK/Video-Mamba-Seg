#!/bin/bash
# Comprehensive Analysis Script
# Run this for complete qualitative analysis

python generate_qualitative_results.py \
    --config configs/qualitative_config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output-dir qualitative_results/comprehensive \
    --split val \
    --max-sequences 25 \
    --device cuda

echo "Comprehensive analysis complete!"
echo "Open qualitative_results/comprehensive/summary_report.html for navigation"
