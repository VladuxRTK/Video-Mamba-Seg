# tests/test_visualization_and_evaluation.py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from utils.visualization import VideoSegmentationVisualizer
from utils.evaluation import DAVISEvaluator

def test_with_untrained_model():
    """Test visualization and evaluation with an untrained model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    
    # Create untrained model
    model = build_model(config).to(device)
    model.eval()
    
    # Create dataloader for a specific sequence
    transform = VideoSequenceAugmentation(
        img_size=(240, 320),
        normalize=True,
        train=False
    )
    
    dataloader = build_davis_dataloader(
        root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
        split='val',
        batch_size=1,
        img_size=(240, 320),
        sequence_length=4,
        specific_sequence="breakdance",  # Test with a specific sequence
        transform=transform
    )
    
    # Initialize visualizer and evaluator
    visualizer = VideoSegmentationVisualizer(save_dir="test_visualization")
    evaluator = DAVISEvaluator()
    
    # Process one batch
    for batch in dataloader:
        # Skip batches without ground truth
        if 'masks' not in batch:
            continue
            
        # Get data
        frames = batch['frames'].to(device)
        masks = batch['masks'].to(device)
        sequence_name = batch['sequence'][0]
        
        print(f"Processing sequence: {sequence_name}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(frames)
        
        # Create visualizations
        visualizer.visualize_sequence(
            frames=frames[0].cpu(),
            pred_masks=outputs['pred_masks'][0].cpu(),
            gt_masks=masks[0].cpu(),
            sequence_name=f"{sequence_name}_untrained"
        )
        
        video_path = visualizer.create_video(
            frames=frames[0].cpu(),
            pred_masks=outputs['pred_masks'][0].cpu(),
            gt_masks=masks[0].cpu(),
            sequence_name=f"{sequence_name}_untrained"
        )
        print(f"Created video visualization at {video_path}")
        
        # Calculate metrics
        metrics = evaluator.evaluate(
        predictions=[outputs['pred_masks'][0].cpu()],
        ground_truths=[masks[0].cpu()],
        sequence_names=[sequence_name]
)
        print("\nMetrics with untrained model:")
        print("\nMetrics with untrained model:")
        def print_metrics(metrics_dict, indent=""):
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    print(f"{indent}{key}:")
                    print_metrics(value, indent + "  ")
                elif isinstance(value, (int, float)):
                    print(f"{indent}{key}: {value:.4f}")
                else:
                    print(f"{indent}{key}: {value}")

        print_metrics(metrics)
        
        # Only process one batch
        break
    
    print("Test completed!")
    return True

if __name__ == "__main__":
    test_with_untrained_model()