import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
from models.video_model import build_model

def test_video_instance_model():
    """Test the video instance segmentation model with random data."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple config
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    
    # Create model and move to device
    model = build_model(config).to(device)
    
    # Create random input data
    batch_size = 1
    sequence_length = 4
    channels = 3
    height = 240
    width = 320
    
    x = torch.randn(batch_size, sequence_length, channels, height, width)
    # Move input to same device as model
    x = x.to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Set model to evaluation mode to avoid batch norm/dropout issues
    model.eval()
    
    # Forward pass with gradient tracking disabled for initial test
    with torch.no_grad():
        try:
            outputs = model(x)
            
            # Check outputs
            assert 'pred_masks' in outputs, "Model should output prediction masks"
            pred_masks = outputs['pred_masks']
            expected_shape = (batch_size, sequence_length, config['num_instances'], height, width)
            print(f"Output pred_masks shape: {pred_masks.shape}")
            assert pred_masks.shape == expected_shape, f"Expected shape {expected_shape}, got {pred_masks.shape}"
            
            print("Basic model test passed!")
        except Exception as e:
            print(f"Error during forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test with gradients if initial test passed
    try:
        # Use a smaller input for gradient test to reduce memory usage
        x_small = torch.randn(1, 2, 3, 120, 160, device=device, requires_grad=True)
        model.train()  # Set to training mode
        outputs = model(x_small)
        pred_masks = outputs['pred_masks']
        loss = pred_masks.mean()  # Simple loss for gradient test
        loss.backward()
        
        print("Gradient test passed!")
        
    except Exception as e:
        print(f"Error during gradient test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_video_instance_model()