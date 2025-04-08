import torch
import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import build_model
from models.temporal_components import InstanceTemporalAttention, InstanceMotionModule

def test_instance_motion_module(model, sample_input):
    """
    Tests the instance motion module's ability to track object movement.
    This test verifies that the module can detect and predict motion between frames.
    """
    print("\nTesting Instance Motion Module...")
    B, T, C, H, W = sample_input.shape
    
    # Extract features using backbone
    with torch.no_grad():
        # Reshape input for backbone
        x = sample_input.view(B * T, C, H, W)
        backbone_features = model.backbone(x)
        last_features = backbone_features[-1]
        
        # Process through motion module
        motion_features, motion_field = model.instance_motion(
            last_features.view(B, T, -1, H, W)
        )
        
        # Verify shapes
        print(f"Motion features shape: {motion_features.shape}")
        print(f"Motion field shape: {motion_field.shape}")
        
        # Check motion field properties
        motion_magnitude = torch.norm(motion_field, dim=2)  # [B, T-1, H, W]
        print(f"Average motion magnitude: {motion_magnitude.mean().item():.4f}")
        print(f"Max motion magnitude: {motion_magnitude.max().item():.4f}")
        
        return motion_features, motion_field

def test_temporal_attention(model, sample_input, motion_field=None):
    """
    Tests the temporal attention mechanism's ability to maintain instance consistency.
    This verifies that the attention module properly tracks instances across frames.
    """
    print("\nTesting Temporal Attention...")
    B, T, C, H, W = sample_input.shape
    
    with torch.no_grad():
        # Get backbone features
        x = sample_input.view(B * T, C, H, W)
        backbone_features = model.backbone(x)
        last_features = backbone_features[-1].view(B, T, -1, H, W)
        
        # Apply temporal attention
        attended_features = model.temporal_attention(last_features, motion_field)
        
        # Analyze attention patterns
        print(f"Attended features shape: {attended_features.shape}")
        
        # Check temporal consistency
        feature_diff = torch.norm(
            attended_features[:, 1:] - attended_features[:, :-1],
            dim=2
        ).mean()
        print(f"Temporal consistency score: {feature_diff.item():.4f}")
        
        return attended_features

def test_temporal_smoothing(model, pred_masks):
    """
    Tests the temporal smoothing module's ability to create consistent instance masks.
    This ensures smooth transitions between frames for each instance.
    """
    print("\nTesting Temporal Smoothing...")
    B, T, N, H, W = pred_masks.shape
    
    with torch.no_grad():
        # Apply temporal smoothing
        smoothed_masks = model.temporal_smooth(pred_masks)
        
        # Check smoothing effect
        original_diff = torch.abs(pred_masks[:, 1:] - pred_masks[:, :-1]).mean()
        smoothed_diff = torch.abs(smoothed_masks[:, 1:] - smoothed_masks[:, :-1]).mean()
        
        print(f"Original temporal difference: {original_diff.item():.4f}")
        print(f"Smoothed temporal difference: {smoothed_diff.item():.4f}")
        
        return smoothed_masks

def visualize_motion_field(motion_field, save_path=None):
    """
    Creates a visualization of the motion field to show instance movement.
    This helps us understand how the model tracks object motion.
    """
    B, T, _, H, W = motion_field.shape
    motion_field = motion_field.cpu().numpy()
    
    fig, axes = plt.subplots(B, T-1, figsize=(4*T, 4*B))
    if B == 1:
        axes = axes[None, :]
    
    for b in range(B):
        for t in range(T-1):
            # Create motion field visualization
            U = motion_field[b, t, 0]
            V = motion_field[b, t, 1]
            
            # Subsample for clearer visualization
            step = 8
            Y, X = np.mgrid[0:H:step, 0:W:step]
            U = U[::step, ::step]
            V = V[::step, ::step]
            
            # Plot motion vectors
            axes[b, t].quiver(X, Y, U, V, scale=1, scale_units='xy')
            axes[b, t].set_title(f'Motion t={t}→{t+1}')
            axes[b, t].axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_temporal_components():
    """
    Comprehensive test of all temporal components working together.
    This test verifies the complete temporal processing pipeline.
    """
    print("\nTesting temporal instance segmentation components...")
    
    # Create test configuration with proper nesting structure
    config = {
        'model': {
            'input_dim': 3,
            'hidden_dims': [32, 64, 128],
            'd_state': 16,
            'temporal_window': 4,
            'dropout': 0.1,
            'mask2former': {
                'hidden_dim': 256,
                'num_queries': 16,
                'nheads': 8,
                'dim_feedforward': 1024,
                'dec_layers': 6,
                'mask_dim': 256
            }
        }
    }
    
    # Create sample input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T = 1, 4  # Batch size and sequence length
    H, W = 240, 320  # Spatial dimensions
    x = torch.randn(B, T, 3, H, W).to(device)
    
    # Extract model config from nested structure
    model_config = config['model']
    
    print("\nInitializing model with configuration:")
    print(f"- Input dimension: {model_config['input_dim']}")
    print(f"- Hidden dimensions: {model_config['hidden_dims']}")
    print(f"- Number of instances: {model_config['mask2former']['num_queries']}")
    
    # Build and initialize model
    try:
        model = build_model(model_config).to(device)
        model.eval()
        
        print("\nTesting complete temporal pipeline:")
        
        # 1. Test Instance Motion Module
        print("\nTesting motion estimation...")
        motion_features, motion_field = test_instance_motion_module(model, x)
        
        # Visualize motion field
        visualize_motion_field(motion_field, "motion_field.png")
        print("Motion field visualization saved to motion_field.png")
        
        # 2. Test Temporal Attention
        print("\nTesting temporal attention...")
        attended_features = test_temporal_attention(model, x, motion_field)
        
        # 3. Full forward pass
        print("\nPerforming full forward pass...")
        with torch.no_grad():
            outputs = model(x)
            
            # Get instance masks
            pred_masks = outputs['pred_masks']
            pred_masks = pred_masks.view(B, T, model.num_instances, H, W)
            
            # 4. Test Temporal Smoothing
            print("\nTesting temporal smoothing...")
            smoothed_masks = test_temporal_smoothing(model, pred_masks)
        
        # Report memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"\nMemory Usage:")
            print(f"- Current: {memory_used:.2f} GB")
            print(f"- Peak: {peak_memory:.2f} GB")
        
        # Verify final output properties
        print("\nFinal Output Properties:")
        print(f"- Instance mask shape: {smoothed_masks.shape}")
        print(f"- Value range: [{smoothed_masks.min():.3f}, {smoothed_masks.max():.3f}]")
        
        print("\nTemporal component test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during temporal testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_temporal_components()

