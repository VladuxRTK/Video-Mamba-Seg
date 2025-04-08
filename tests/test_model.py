import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import VideoMambaSegmentation, build_model

def create_sample_config():
    """Creates a sample configuration for testing the model.
    
    The configuration includes all necessary parameters for both the backbone
    and decoder components, ensuring proper initialization of the full pipeline.
    """
    return {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'dropout': 0.1,
        'd_conv': 4,
        'expand': 2,
        'num_classes': 21,
        'mask2former': {
            'hidden_dim': 256,
            'num_queries': 100,
            'nheads': 8,
            'dim_feedforward': 1024,
            'dec_layers': 6,
            'mask_dim': 256,
            'enforce_input_project': False
        }
    }

def test_model_initialization():
    """Tests if the model initializes correctly with the sample configuration."""
    print("\nTesting model initialization...")
    
    config = create_sample_config()
    model = build_model(config)
    
    assert hasattr(model, 'backbone'), "Backbone is missing"
    assert hasattr(model, 'seg_head'), "Segmentation head is missing"
    
    print("Model initialization test passed!")


def test_model_forward():
    """Tests the model's forward pass with proper tensor shape handling."""
    print("\nTesting model forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = create_sample_config()
    model = build_model(config).to(device)
    model.eval()
    
    # Create test data
    batch_size = 2
    time_steps = 2
    channels = 3
    height = 64
    width = 64
    
    # Create input tensors with proper shapes
    dummy_input = torch.randn(batch_size, time_steps, channels, height, width).to(device)
    mask_features = torch.randn(batch_size, channels, time_steps, height, width).to(device)
    
    try:
        print("Testing inference mode...")
        with torch.no_grad():
            # Forward pass with both inputs
            outputs = model(dummy_input, mask_features)  # No keyword argument
            
            # Verify outputs
            assert 'pred_masks' in outputs, "pred_masks missing from outputs"
            pred_masks = outputs['pred_masks']
            
            # Expected shape after temporal flattening
            expected_shape = (batch_size * time_steps, config['num_classes'], height, width)
            print(f"Expected shape: {expected_shape}")
            print(f"Actual shape: {pred_masks.shape}")
            
            assert pred_masks.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {pred_masks.shape}"
            )
            
            # Verify output values
            assert torch.isfinite(pred_masks).all(), "Output contains inf or nan values"
            
            print("Forward pass test completed successfully!")
            return True
            
    except Exception as e:
        print(f"\nError during forward pass test: {str(e)}")
        return False

def test_model_output_values():
    """Tests if the model's outputs have reasonable values within expected ranges."""
    print("\nTesting model output values...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_sample_config()
    model = build_model(config).to(device)
    model.eval()
    
    batch_size = 2
    time_steps = 2
    dummy_input = torch.randn(batch_size, time_steps, 3, 64, 64).to(device)
    mask_features = torch.randn(batch_size, 3, time_steps, 64, 64).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input, mask_features=mask_features)
        masks = outputs['pred_masks']
        
        # Verify values are within reasonable range (e.g., post-sigmoid would be 0-1)
        assert torch.all((masks >= 0) & (masks <= 1)), "Mask values outside expected range"
        print("Output values test passed!")
        return True

def run_all_tests():
    """Runs all model tests in sequence with proper error handling."""
    try:
        test_model_initialization()
        test_model_forward()
        test_model_output_values()
        print("\nAll tests passed successfully!")
        return True
    except Exception as e:
        print(f"\nTest suite failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    run_all_tests()