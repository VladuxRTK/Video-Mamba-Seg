import torch
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import build_model

def create_test_config(small_test=False):
    """
    Creates a configuration for testing. Can create either a full-size or small test configuration.
    
    Args:
        small_test: If True, creates a smaller configuration for dimension testing
    """
    if small_test:
        return {
            'model': {
                'input_dim': 3,
                'hidden_dims': [16, 32, 64],  # Smaller for testing
                'd_state': 16,
                'temporal_window': 4,
                'dropout': 0.1,
                'd_conv': 4,
                'expand': 2,
                'mask2former': {
                    'hidden_dim': 128,
                    'num_queries': 16,
                    'nheads': 4,
                    'dim_feedforward': 256,
                    'dec_layers': 3,
                    'mask_dim': 128,
                    'enforce_input_project': False
                }
            }
        }
    else:
        return {
            'model': {
                'input_dim': 3,
                'hidden_dims': [32, 64, 128],
                'd_state': 16,
                'temporal_window': 4,
                'dropout': 0.1,
                'd_conv': 4,
                'expand': 2,
                'mask2former': {
                    'hidden_dim': 256,
                    'num_queries': 16,
                    'nheads': 4,
                    'dim_feedforward': 512,
                    'dec_layers': 6,
                    'mask_dim': 256,
                    'enforce_input_project': False
                }
            }
        }

def test_memory_usage(device):
    """Print current GPU memory usage if using CUDA."""
    if device.type == 'cuda':
        current_mem = torch.cuda.memory_allocated() / 1e9
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Current GPU memory: {current_mem:.2f} GB")
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

def test_dimensions():
    """
    Test if dimensions are correct throughout the pipeline.
    This function uses a smaller model and input size to verify dimension handling.
    """
    print("\nStarting dimension testing...")
    
    # Create small test input
    B, T, C = 1, 4, 3
    H, W = 128, 128
    x = torch.randn(B, T, C, H, W)
    
    # Use smaller configuration for dimension testing
    config = create_test_config(small_test=True)
    
    try:
        # Create and move model to available device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(config).to(device)
        model.eval()
        x = x.to(device)
        
        print(f"\nTesting with configuration:")
        print(f"- Input shape: {x.shape}")
        print(f"- Hidden dimensions: {config['model']['hidden_dims']}")
        print(f"- Number of queries: {config['model']['mask2former']['num_queries']}")
        
        with torch.no_grad():
            # Process through model
            outputs = model(x)
            pred_masks = outputs['pred_masks']
            
            # Verify output dimensions
            expected_instances = config['model']['mask2former']['num_queries']
            B_out, N_out, H_out, W_out = pred_masks.shape
            
            print("\nOutput dimensions:")
            print(f"- Batch size (B*T): {B_out} (expected {B*T})")
            print(f"- Number of instances: {N_out} (expected {expected_instances})")
            print(f"- Height: {H_out} (expected {H})")
            print(f"- Width: {W_out} (expected {W})")
            
            # Verify dimension relationships
            assert B_out == B * T, "Batch dimension mismatch"
            assert N_out == expected_instances, "Instance dimension mismatch"
            assert H_out == H, "Height mismatch"
            assert W_out == W, "Width mismatch"
            
            print("\nDimension test passed successfully!")
            return True
            
    except Exception as e:
        print(f"\nDimension test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_instance_segmentation():
    """
    Tests the complete instance segmentation pipeline with full-size configuration.
    """
    print("\nTesting instance segmentation with temporal components...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create model with full-size config
        config = create_test_config(small_test=False)
        print("\nCreating model...")
        model = build_model(config).to(device)
        model.eval()
        test_memory_usage(device)
        
        # Create test dimensions
        batch_size = 1
        sequence_length = 4
        height = 240
        width = 320
        
        print(f"\nTest dimensions:")
        print(f"- Batch size: {batch_size}")
        print(f"- Sequence length: {sequence_length}")
        print(f"- Resolution: {height}x{width}")
        print(f"- Number of queries: {config['model']['mask2former']['num_queries']}")
        
        # Create input tensor
        print("\nCreating input tensor...")
        video_input = torch.randn(
            batch_size, sequence_length, 3, height, width
        ).to(device)
        test_memory_usage(device)
        
        # Run forward pass
        print("\nRunning forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(video_input)
        
        inference_time = time.time() - start_time
        fps = sequence_length / inference_time
        
        print(f"\nPerformance metrics:")
        print(f"- Inference time: {inference_time:.3f} seconds")
        print(f"- Frames per second: {fps:.2f}")
        test_memory_usage(device)
        
        # Verify outputs
        print("\nVerifying outputs...")
        pred_masks = outputs['pred_masks']
        print(f"- Prediction shape: {pred_masks.shape}")
        print(f"- Value range: [{pred_masks.min():.3f}, {pred_masks.max():.3f}]")
        print(f"- Memory after forward pass:")
        test_memory_usage(device)
        
        print("\nInstance segmentation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First run dimension test with smaller configuration
    if not test_dimensions():
        print("\nDimension test failed - skipping full model test")
        sys.exit(1)
        
    # If dimensions are correct, run full model test
    print("\nDimension test passed - proceeding with full model test")
    test_instance_segmentation()