import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLayerNorm(nn.Module):
    """
    Custom LayerNorm that handles temporal data while maintaining dimensional consistency.
    This ensures proper normalization across channels while preserving temporal information.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        # x input shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Rearrange for channel normalization
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        x = x.reshape(-1, C)  # Combine all dimensions except channels
        
        # Apply normalization
        x = self.norm(x)
        
        # Restore original shape
        x = x.view(B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)  # Back to [B, C, T, H, W]
        return x

class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Spatial processing
        self.spatial_proj = nn.Conv2d(d_model, d_model, 1)
        
        # Temporal processing
        self.temporal_proj = nn.Conv3d(
            d_model, d_model,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        
        # Here's the key change: we're using d_model in both state projections
        # because our features are in d_model dimension space
        self.state_proj = nn.Linear(d_state, d_model)
        # This projection now goes from d_model to d_state directly
        self.state_update = nn.Linear(d_model, d_state)
        
    def forward(self, x, state=None):
        batch_size, time_steps, channels, height, width = x.shape
        
        if state is None:
            state = self.init_state(batch_size).to(x.device)
        
        # Project state
        state_features = self.state_proj(state)  # [B, d_model]
        state_features = state_features.view(batch_size, self.d_model, 1, 1, 1)
        state_features = state_features.expand(-1, -1, time_steps, height, width)
        
        # Process temporal dimension
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        temporal_features = self.temporal_proj(x)
        
        # Process each timestep
        output = []
        for t in range(time_steps):
            curr_features = temporal_features[:, :, t]  # [B, C, H, W]
            spatial_features = self.spatial_proj(curr_features)
            combined = spatial_features + state_features[:, :, t]
            output.append(combined)
        
        output = torch.stack(output, dim=1)  # [B, T, C, H, W]
        
        # Here's the critical fix for dimension mismatch:
        features_pooled = output.mean([-2, -1])  # Average over spatial dimensions [B, T, C]
        features_mean = features_pooled.mean(1)   # Average over temporal dimension [B, C]
        new_state = self.state_update(features_mean)  # Project from d_model to d_state [B, d_state]
        
        return output, new_state

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.d_state)
        
class TemporalCNNBackbone(nn.Module):
    """
    CNN backbone that properly handles temporal information and multi-scale features
    while maintaining dimensional consistency throughout the network.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        
        for dim in hidden_dims:
            self.layers.append(nn.Sequential(
                # 3D convolution for spatiotemporal processing
                nn.Conv3d(
                    curr_dim, dim,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1)
                ),
                TemporalLayerNorm(dim),
                nn.ReLU(inplace=True)
            ))
            curr_dim = dim
            
    def forward(self, x):
        """
        Forward pass that maintains temporal dimension throughout processing
        Args:
            x: Input tensor [B, T, C, H, W]
        Returns:
            List of features at different scales, each [B, T, C', H, W]
        """
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        features = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            # Restore temporal dimension order
            features.append(current.permute(0, 2, 1, 3, 4))
            
        return features

def test_simple_backbone():
    print("Starting backbone test...")
    
    # Create test data
    batch_size = 2
    time_steps = 3
    channels = 3
    height = 64
    width = 64
    
    # Create input video
    video = torch.randn(batch_size, time_steps, channels, height, width)
    print(f"\nInput video shape: {video.shape}")
    
    # Initialize models
    hidden_dims = [32, 64, 128]
    temporal_cnn = TemporalCNNBackbone(channels, hidden_dims)
    mamba_blocks = nn.ModuleList([
        SimpleMambaBlock(dim, d_state=16)
        for dim in hidden_dims
    ])
    
    print("\nProcessing video through backbone...")
    
    try:
        # Process through CNN
        cnn_features = temporal_cnn(video)
        
        # Process through Mamba blocks
        final_features = []
        states = [None] * len(mamba_blocks)
        
        for i, (feat, mamba) in enumerate(zip(cnn_features, mamba_blocks)):
            mamba_out, new_state = mamba(feat, states[i])
            states[i] = new_state
            final_features.append(mamba_out)
        
        # Print output shapes
        print("\nOutput feature shapes at each scale:")
        for i, feat in enumerate(final_features):
            print(f"Scale {i + 1}: {feat.shape}")
        
        print("\nTest completed successfully!")
        return final_features
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print(f"Error type: {type(e)}")
        raise e

if __name__ == "__main__":
    features = test_simple_backbone()