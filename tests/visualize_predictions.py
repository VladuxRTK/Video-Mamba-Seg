import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation

def visualize_predictions():
    """Visualize model predictions on DAVIS data."""
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
            
            # Visualize results
            B, T, N, H, W = pred_masks.shape
            
            # Create visualization figure
            fig, axes = plt.subplots(T, 3, figsize=(15, 5*T))
            
            for t in range(T):
                # Original frame
                frame = frames[0, t].permute(1, 2, 0).numpy()  # [H, W, C]
                # Denormalize if needed
                if frame.max() <= 1.0:
                    frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
                axes[t, 0].imshow(frame)
                axes[t, 0].set_title(f"Frame {t}")
                
                # Ground truth mask
                if gt_masks is not None:
                    gt = gt_masks[0, t].numpy()
                    axes[t, 1].imshow(gt, cmap='tab20')
                    axes[t, 1].set_title(f"Ground Truth {t}")
                else:
                    axes[t, 1].axis('off')
                
                # Predicted masks
                # Combine instance masks by taking argmax
                pred_combined = torch.zeros((H, W))
                for n in range(N):
                    mask = pred_masks[0, t, n] > 0.5
                    # Assign instance ID (add 1 to avoid conflict with background)
                    pred_combined[mask.squeeze()] = n + 1
                
                axes[t, 2].imshow(pred_combined.numpy(), cmap='tab20')
                axes[t, 2].set_title(f"Prediction {t}")
            
            plt.tight_layout()
            plt.savefig("prediction_visualization.png")
            print(f"Visualization saved to prediction_visualization.png")
            
            # Only process one batch
            break
            
        return True
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    visualize_predictions()