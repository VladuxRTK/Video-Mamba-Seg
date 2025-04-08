import torch
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from models.model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from metrics.evaluator import DAVISEvaluator

def setup_logging(save_dir: Path):
    """Configure logging to both file and console for evaluation results."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )

# Replace the existing save_visualization function with your improved visualizer
def save_visualization(
    frames: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    sequence_name: str,
    save_dir: Path
):
    """
    Save visualization using the enhanced visualization tools.
    """
    from utils.visualization import VideoSegmentationVisualizer
    
    # Create visualizer with the specified save directory
    visualizer = VideoSegmentationVisualizer(save_dir=save_dir)
    
    # Create sequence visualization
    visualizer.visualize_sequence(
        frames=frames,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        sequence_name=sequence_name
    )
    
    # Create video visualization
    video_path = visualizer.create_video(
        frames=frames,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        sequence_name=sequence_name
    )
    logging.info(f"Created video visualization at {video_path}")
    
    # Create analysis dashboard
    visualizer.create_analysis_dashboard(
        frames=frames,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        sequence_name=sequence_name
    )

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    evaluator: DAVISEvaluator,
    device: torch.device,
    save_viz: bool = False,
    viz_dir: Optional[Path] = None
) -> dict:
    """
    Evaluate model on the dataset with progress tracking and comprehensive metrics.
    Returns a dictionary containing all computed metrics.
    """
    model.eval()
    all_metrics = []
    
    # Create progress bar for evaluation
    pbar = tqdm(dataloader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in pbar:
            # Move data to device
            frames = batch['frames'].to(device)
            masks = batch['masks'].to(device)
            sequence = batch['sequence'][0]  # Assuming batch_size=1
            
            # Forward pass
            outputs = model(frames)
            pred_masks = outputs['pred_masks']
            
            # Compute metrics for this sequence
            metrics = evaluator.evaluate_sequence(
                pred_masks[0],  # Remove batch dimension
                masks[0]
            )
            all_metrics.append(metrics)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'J_mean': f"{metrics['J_mean']:.4f}",
                'F_mean': f"{metrics.get('F_mean', 0):.4f}"
            })
            
            # Save visualizations if requested
            if save_viz and viz_dir is not None:
                save_visualization(
                    frames[0],
                    pred_masks[0],
                    masks[0],
                    sequence,
                    viz_dir
                )
    
    # Compute final metrics
    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = np.mean(values)
    
    return final_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate video segmentation model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization of predictions')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup paths and logging
    save_dir = Path(config['paths']['checkpoints'])
    setup_logging(save_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Build model and load checkpoint
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create data transform (no augmentation during evaluation)
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        normalize=True,
        train=False
    )
    
    # Create dataloader
    dataloader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split=args.split,
        batch_size=1,  # Use batch size 1 for evaluation
        transform=transform,
        **{k: v for k, v in config['dataset'].items() if k != 'batch_size'}
    )
    
    # Initialize evaluator
    evaluator = DAVISEvaluator()
    
    # Setup visualization directory if needed
    viz_dir = None
    if args.save_viz:
        viz_dir = Path(config['paths']['visualizations']) / args.split
        viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving visualizations to {viz_dir}")
    
    # Run evaluation
    logging.info(f"Evaluating model on {args.split} split...")
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        evaluator=evaluator,
        device=device,
        save_viz=args.save_viz,
        viz_dir=viz_dir
    )
    
    # Log results
    logging.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save metrics to file
    results_file = save_dir / f'metrics_{args.split}.txt'
    with open(results_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    logging.info(f"\nSaved metrics to {results_file}")

if __name__ == '__main__':
    main()