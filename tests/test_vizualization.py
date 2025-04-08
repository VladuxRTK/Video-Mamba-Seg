import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from losses.video_instance_loss import VideoInstanceSegmentationLoss

# Import our new components
from utils.visualization import VideoSegmentationVisualizer, visualize_instance_tracking
from utils.evaluation import VideoInstanceEvaluator, DAVISEvaluator

def test_visualization_tools():
    """Test the enhanced visualization tools."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    model = build_model(config).to(device)
    model.eval()  # Set to evaluation mode
    
    try:
        # Create dataloader
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
            specific_sequence="breakdance",
            transform=transform
        )
        
        # Create visualizer
        visualizer = VideoSegmentationVisualizer(save_dir="visualization_output")
        
        # Process one batch
        for batch in dataloader:
            # Move data to device
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            gt_masks = batch.get('masks')
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)  # [B, T, H, W]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(frames)
            
            pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
            
            # Move tensors to CPU for visualization
            frames = frames.cpu()
            pred_masks = pred_masks.cpu()
            if gt_masks is not None:
                gt_masks = gt_masks.cpu()
            
            # Test single frame visualization
            fig = visualizer.visualize_frame(
                frame=frames[0, 0],
                pred_masks=pred_masks[0, 0],
                gt_mask=gt_masks[0, 0] if gt_masks is not None else None,
                frame_idx=0,
                title="Test Frame Visualization"
            )
            plt.close(fig)
            
            # Test sequence visualization
            sequence_name = batch['sequence'][0]
            _ = visualizer.visualize_sequence(
                frames=frames[0],
                pred_masks=pred_masks[0],
                gt_masks=gt_masks[0] if gt_masks is not None else None,
                sequence_name=sequence_name
            )
            
            # Test video creation
            video_path = visualizer.create_video(
                frames=frames[0],
                pred_masks=pred_masks[0],
                gt_masks=gt_masks[0] if gt_masks is not None else None,
                sequence_name=sequence_name
            )
            print(f"Video saved to: {video_path}")
            
            # Test dashboard
            dashboard = visualizer.create_analysis_dashboard(
                frames=frames[0],
                pred_masks=pred_masks[0],
                gt_masks=gt_masks[0] if gt_masks is not None else None,
                metrics={'J&F': 0.5, 'J_mean': 0.6, 'F_mean': 0.4, 'T_mean': 0.8},
                sequence_name=sequence_name
            )
            plt.close(dashboard)
            
            # Test instance tracking visualization
            track_path = visualize_instance_tracking(
                frames=frames[0],
                pred_masks=pred_masks[0],
                sequence_name=sequence_name,
                save_dir="visualization_output"
            )
            print(f"Instance tracking visualization saved to: {track_path}")
            
            # Only process one batch
            break
            
        print("Visualization tools test completed!")
        return True
        
    except Exception as e:
        print(f"Error during visualization test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

