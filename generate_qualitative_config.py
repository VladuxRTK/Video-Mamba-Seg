#!/usr/bin/env python3
"""
Configuration and Setup Helper for Qualitative Results Generation

This script helps you set up and configure the qualitative results generation
with your specific model and dataset paths.
"""

import yaml
import argparse
from pathlib import Path
import json

def create_qualitative_config():
    """Create a configuration file for qualitative results generation."""
    
    config = {
        'model': {
            'input_dim': 3,
            'hidden_dims': [32, 64, 128],
            'd_state': 16,
            'temporal_window': 4,
            'dropout': 0.1,
            'd_conv': 4,
            'expand': 2
        },
        'dataset': {
            'img_size': [240, 320],
            'sequence_length': 3,
            'sequence_stride': 2,
            'num_workers': 4
        },
        'paths': {
            'davis_root': '/mnt/c/Datasets/DAVIS',  # Update this path
            'checkpoints': 'checkpoints',
            'visualizations': 'visualizations'
        },
        'qualitative': {
            'max_sequences_quick': 6,
            'max_sequences_full': 20,
            'output_formats': ['png', 'pdf'],
            'video_fps': 10,
            'figure_dpi': 300,
            'colors': {
                'prediction': [0, 255, 0],      # Green
                'ground_truth': [255, 0, 0],    # Red  
                'overlap': [255, 255, 0],       # Yellow
                'background': [128, 128, 128]   # Gray
            }
        }
    }
    
    return config

def setup_project_structure():
    """Set up the recommended project structure for qualitative results."""
    
    directories = [
        'checkpoints',
        'configs',
        'qualitative_results',
        'qualitative_results/paper_figures',
        'qualitative_results/presentations', 
        'qualitative_results/quick_demos',
        'qualitative_results/comprehensive',
        'logs',
        'visualizations'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_example_scripts():
    """Create example scripts for different use cases."""
    
    # Quick demo script
    quick_demo_script = """#!/bin/bash
# Quick Demo Script
# Run this for a fast visual check of your model

python quick_qualitative_demo.py \\
    --checkpoint checkpoints/best_model.pth \\
    --config configs/qualitative_config.yaml \\
    --output-dir qualitative_results/quick_demo \\
    --split val \\
    --max-sequences 6 \\
    --device cuda

echo "Quick demo complete! Check qualitative_results/quick_demo/"
"""
    
    # Paper figures script
    paper_script = """#!/bin/bash
# Paper Figures Script
# Run this to generate publication-ready figures

python generate_qualitative_results.py \\
    --config configs/qualitative_config.yaml \\
    --checkpoint checkpoints/best_model.pth \\
    --output-dir qualitative_results/paper_figures \\
    --split val \\
    --max-sequences 15 \\
    --device cuda

echo "Paper figures complete! Check qualitative_results/paper_figures/"
echo "Main figure: qualitative_results/paper_figures/paper_figures/qualitative_comparison.pdf"
"""

    # Comprehensive analysis script
    comprehensive_script = """#!/bin/bash
# Comprehensive Analysis Script
# Run this for complete qualitative analysis

python generate_qualitative_results.py \\
    --config configs/qualitative_config.yaml \\
    --checkpoint checkpoints/best_model.pth \\
    --output-dir qualitative_results/comprehensive \\
    --split val \\
    --max-sequences 25 \\
    --device cuda

echo "Comprehensive analysis complete!"
echo "Open qualitative_results/comprehensive/summary_report.html for navigation"
"""
    
    scripts = {
        'quick_demo.sh': quick_demo_script,
        'generate_paper_figures.sh': paper_script,
        'comprehensive_analysis.sh': comprehensive_script
    }
    
    for script_name, script_content in scripts.items():
        script_path = Path(script_name)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)  # Make executable
        print(f"Created script: {script_name}")

def create_readme():
    """Create a README file with usage instructions."""
    
    readme_content = """# Qualitative Results Generation

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
python generate_qualitative_results.py \\
    --config my_config.yaml \\
    --checkpoint my_model.pth \\
    --output-dir my_results \\
    --split test \\
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
"""
    
    with open('README_qualitative.md', 'w') as f:
        f.write(readme_content)
    print("Created README_qualitative.md")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Set up qualitative results generation')
    parser.add_argument('--davis-path', type=str, default='/mnt/c/Datasets/DAVIS',
                       help='Path to DAVIS dataset')
    parser.add_argument('--setup-structure', action='store_true',
                       help='Create recommended directory structure')
    parser.add_argument('--create-scripts', action='store_true',
                       help='Create example scripts')
    parser.add_argument('--create-readme', action='store_true',
                       help='Create README file')
    parser.add_argument('--all', action='store_true',
                       help='Set up everything')
    
    args = parser.parse_args()
    
    if args.all:
        args.setup_structure = True
        args.create_scripts = True
        args.create_readme = True
    
    print("Setting up qualitative results generation...")
    
    # Create configuration file
    config = create_qualitative_config()
    config['paths']['davis_root'] = args.davis_path
    
    Path('configs').mkdir(exist_ok=True)
    with open('configs/qualitative_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("Created configs/qualitative_config.yaml")
    
    if args.setup_structure:
        setup_project_structure()
    
    if args.create_scripts:
        create_example_scripts()
    
    if args.create_readme:
        create_readme()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Update the DAVIS dataset path in configs/qualitative_config.yaml")
    print("2. Place your trained model checkpoint in checkpoints/")
    print("3. Run ./quick_demo.sh for a quick test")
    print("4. Check README_qualitative.md for detailed usage instructions")

if __name__ == '__main__':
    main()