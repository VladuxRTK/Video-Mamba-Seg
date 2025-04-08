import torch
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import build_model

def create_realistic_config():
    """Creates configuration with memory-efficient parameters."""
    return {
        'model': {  # Nest model parameters under 'model' key
            'input_dim': 3,
            'hidden_dims': [32, 64, 128],  # Reduced channel dimensions
            'd_state': 16,
            'temporal_window': 4,          # Reduced temporal window
            'dropout': 0.1,
            'd_conv': 4,
            'expand': 2,
            'num_classes': 21,
            'mask2former': {
                'hidden_dim': 256,
                'num_queries': 100,
                'nheads': 8,
                'dim_feedforward': 1024,    # Reduced feedforward dimension
                'dec_layers': 6,
                'mask_dim': 256,
                'enforce_input_project': False
            }
        },
        # Additional configuration sections
        'training': {
            'epochs': 100,
            'batch_size': 1,
            'mixed_precision': True
        },
        'dataset': {
            'img_size': [240, 320],  # Half resolution for testing
            'sequence_length': 4
        }
    }

def test_realistic_scenario():
    """Tests the model with memory-efficient but still realistic parameters."""
    print("\nTesting memory-efficient video segmentation scenario...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with efficient config
    config = create_realistic_config()
    model = build_model(config).to(device)
    model.eval()
    
    # Memory-efficient dimensions but still realistic ratio
    batch_size = 1          # Process 1 video at a time
    sequence_length = 4     # 4 frames per sequence
    height = config['dataset']['img_size'][0]  # Use height from config
    width = config['dataset']['img_size'][1]   # Use width from config
    channels = config['model']['input_dim']    # Use channels from config
    
    try:
        # Create input tensors
        print("\nPreparing input tensors...")
        print(f"Input dimensions:")
        print(f"- Batch size: {batch_size} (videos)")
        print(f"- Sequence length: {sequence_length} frames")
        print(f"- Resolution: {height}x{width}")
        print(f"- Channels: {channels} (RGB)")
        
        # Simulate video input
        video_input = torch.randn(batch_size, sequence_length, channels, height, width).to(device)
        
        # Simulate mask features
        mask_features = torch.randn(batch_size, channels, sequence_length, height, width).to(device)
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            print(f"\nInitial GPU memory used: {start_mem/1e9:.2f} GB")
        
        # Time the forward pass
        print("\nRunning forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(video_input, mask_features)
        
        # Calculate timing
        inference_time = time.time() - start_time
        
        # Memory statistics
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            mem_diff = peak_mem - start_mem
            print(f"\nMemory Usage:")
            print(f"- Peak memory: {peak_mem/1e9:.2f} GB")
            print(f"- Memory increase: {mem_diff/1e9:.2f} GB")
        
        # Performance metrics
        fps = (batch_size * sequence_length)/inference_time
        print(f"\nPerformance Metrics:")
        print(f"- Total inference time: {inference_time:.3f} seconds")
        print(f"- Frames per second: {fps:.2f}")
        
        # Analyze outputs
        pred_masks = outputs['pred_masks']
        print(f"\nOutput Analysis:")
        print(f"- Prediction shape: {pred_masks.shape}")
        
        print("\nMemory-efficient scenario test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        return False

if __name__ == "__main__":
    test_realistic_scenario()