# Qualitative Results Generation

This directory contains tools for generating qualitative results from your video segmentation model.

## Quick Start

1. **Setup**: Update the DAVIS dataset path in `configs/qualitative_config.yaml`
2. **Quick Demo**: Run `./quick_demo.sh` for a fast visual check
3. **Paper Figures**: Run `./generate_paper_figures.sh` for publication-ready figures
4. **Full Analysis**: Run `./comprehensive_analysis.sh` for complete analysis

## Available Scripts

### Quick Demo (`quick_qualitative_demo.py`)
- Fast generation of basic visualizations
- Grid overview of multiple sequences
- Failure case analysis
- Perfect for debugging and quick checks

### Comprehensive Generation (`generate_qualitative_results.py`)
- Publication-quality figures
- Temporal consistency analysis
- Error analysis plots
- Video outputs
- Presentation slides
- Detailed HTML summary

### Batch Script (`generate_results.sh`)
- Easy-to-use wrapper script
- Multiple generation modes
- Automatic file organization

## Output Structure

```
qualitative_results/
├── quick_demo/
│   ├── grid_overview.png
│   ├── failure_cases.png
│   └── *_comparison.png
├── paper_figures/
│   ├── paper_figures/
│   │   ├── qualitative_comparison.png
│   │   └── qualitative_comparison.pdf
│   └── error_analysis/
│       └── comprehensive_error_analysis.png
├── comprehensive/
│   ├── summary_report.html
│   ├── paper_figures/
│   ├── videos/
│   ├── presentation_slides/
│   └── individual_frames/
└── presentations/
    ├── slides/
    └── videos/
```

## Usage Examples

### Basic Usage
```bash
# Quick visual check
python quick_qualitative_demo.py --checkpoint checkpoints/best_model.pth

# Paper figures only
python generate_qualitative_results.py --checkpoint checkpoints/best_model.pth --max-sequences 10

# Full analysis
./comprehensive_analysis.sh
```

### Advanced Usage
```bash
# Custom configuration
python generate_qualitative_results.py \
    --config my_config.yaml \
    --checkpoint my_model.pth \
    --output-dir my_results \
    --split test \
    --specific-sequence blackswan

# Using the batch script
./generate_results.sh --mode paper --checkpoint checkpoints/best_model.pth --max-sequences 15
```

## Configuration

Edit `configs/qualitative_config.yaml` to customize:
- Dataset paths
- Model parameters  
- Visualization settings
- Output formats

## Requirements

- PyTorch
- matplotlib
- opencv-python
- tqdm
- pyyaml
- numpy

## Tips

1. Start with `quick_demo.sh` to verify everything works
2. Use `--max-sequences 5` for fast iteration during development
3. For papers, generate both PNG and PDF versions of figures
4. The HTML summary report provides easy navigation of all results
5. Videos are great for presentations and supplementary materials

## Troubleshooting

- **CUDA out of memory**: Reduce `max_sequences` or use `--device cpu`
- **Missing dataset**: Update `davis_root` path in config
- **Import errors**: Ensure all project modules are in PYTHONPATH
- **Low quality results**: Check model checkpoint and training logs
