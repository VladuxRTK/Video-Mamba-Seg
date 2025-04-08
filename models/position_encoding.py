import torch
import torch.nn as nn
import math

class LeveledPositionEmbeddingSine(nn.Module):
    """
    Position encoding that exactly matches feature dimensions at each level.
    The encoding dimension will precisely match the input feature dimension,
    ensuring compatibility throughout the feature hierarchy.
    """
    def __init__(self, feature_dim: int, temperature: int = 10000):
        super().__init__()
        # The feature dimension determines the number of position encoding channels
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Number of position encoding features matches input exactly
        self.num_pos_feats = feature_dim
        
        print(f"Initializing position embedding with {feature_dim} channels")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate position encodings that exactly match input dimensions.
        
        Args:
            x: Input tensor [B, C, H, W] where C is feature_dim
            
        Returns:
            Position encodings [B, C, H, W] with same channel count as input
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape: {x.shape}")
        
        if x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Input has {x.shape[1]} channels but position encoding "
                f"was initialized for {self.feature_dim} channels"
            )
        
        # Generate normalized coordinate grids
        not_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        
        # Create position encoding with correct number of channels
        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Stack sin and cos embeddings
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        
        # Combine x and y embeddings to get final position encoding
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Double check output dimensions
        assert pos.shape == x.shape, (
            f"Position encoding shape {pos.shape} doesn't match input shape {x.shape}"
        )
        
        return pos