#!/bin/bash
# Paper Figures Script
# Run this to generate publication-ready figures

python generate_qualitative_results.py \
    --config configs/qualitative_config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output-dir qualitative_results/paper_figures \
    --split val \
    --max-sequences 15 \
    --device cuda

echo "Paper figures complete! Check qualitative_results/paper_figures/"
echo "Main figure: qualitative_results/paper_figures/paper_figures/qualitative_comparison.pdf"
