import torch
import sys
from pathlib import Path

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Update imports to match actual implementation
from models.decoder import (
    SegmentationHead,
    MambaMask2FormerDecoder,
    EnhancedTemporalSmoothingModule
)

def test_decoder():
    """
    Tests the complete decoder pipeline including Mask2Former integration,
    temporal smoothing, and final mask generation.
    """
    print("Testing decoder with dummy data...")
    
    # Set up device for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data with realistic dimensions
    batch_size = 2
    time_steps = 2
    channels = [32, 64, 128]
    height = 64
    width = 64
    num_classes = 21
    
    # Create multi-scale features
    features = [
        torch.randn(batch_size * time_steps, c, height, width).to(device)
        for c in channels
    ]
    
    # Create optical flow between consecutive frames
    flows = torch.randn(batch_size, time_steps-1, 2, height, width).to(device)
    
    # Initialize decoder
    mask2former_config = {
        'hidden_dim': 256,
        'num_queries': 100,
        'nheads': 8,
        'dim_feedforward': 1024,
        'dec_layers': 6,
        'mask_dim': 256,
        'enforce_input_project': False
    }
    
    try:
        decoder = SegmentationHead(
            in_channels=channels,
            mask2former_config=mask2former_config,
            num_classes=num_classes
        ).to(device)
        
        # Forward pass
        outputs = decoder(features, flows)
        
        # Verify output is dictionary during inference
        assert isinstance(outputs, dict), "Output should be a dictionary"
        assert 'pred_masks' in outputs, "Output should contain 'pred_masks'"
        
        masks = outputs['pred_masks']
        print(f"\nOutput mask shape: {masks.shape}")
        
        # Verify output dimensions
        expected_shape = (batch_size * time_steps, num_classes, height, width)
        assert masks.shape == expected_shape, f"Expected shape {expected_shape}, got {masks.shape}"
        
        print("\nDecoder test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during decoder test: {str(e)}")
        raise e  # Re-raise to see full traceback

if __name__ == "__main__":
    test_decoder()