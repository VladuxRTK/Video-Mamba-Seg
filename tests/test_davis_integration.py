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
from losses.video_instance_loss import VideoInstanceSegmentationLoss

def test_davis_integration():
    """Test model with actual DAVIS data."""
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
    
    # Create dataloader with a single specific sequence
    transform = VideoSequenceAugmentation(
        img_size=(240, 320),
        normalize=True,
        train=False
    )
    
    try:
        dataloader = build_davis_dataloader(
            root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
            split='val',
            batch_size=1,
            img_size=(240, 320),
            sequence_length=4,
            specific_sequence="breakdance",  # Test with a specific sequence
            transform=transform
        )
        
        # Create loss function
        criterion = VideoInstanceSegmentationLoss()
        
        # Process one batch
        for batch in dataloader:
            # Move data to device
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            masks = batch.get('masks')
            if masks is not None:
                masks = masks.to(device)  # [B, T, H, W]
            
            print(f"Input frames shape: {frames.shape}")
            if masks is not None:
                print(f"Ground truth masks shape: {masks.shape}")
            
            # Forward pass with gradient tracking disabled
            with torch.no_grad():
                outputs = model(frames)
            
            print(f"Output pred_masks shape: {outputs['pred_masks'].shape}")
            
            # Compute loss if ground truth available
            if masks is not None:
                loss_dict = criterion(outputs, {'masks': masks})
                print(f"Loss values: {loss_dict}")
            
            # Only process one batch
            break
        
        print("DAVIS integration test passed!")
        return True
        
    except Exception as e:
        print(f"Error during DAVIS integration test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_davis_integration()