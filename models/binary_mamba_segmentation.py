import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from collections import deque

from mamba_ssm import Mamba as MambaBlock

class VideoMambaBlock(nn.Module):
    """Efficient Mamba block for video processing with spatial reshaping."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize Mamba block - the core innovation
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape spatial dimensions into sequence length."""
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
    
    def _restore_spatial(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Restore spatial dimensions."""
        b, hw, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba with spatial reshaping."""
        # Store original shape
        b, c, h, w = x.shape

        # Prepare input for Mamba
        x_seq = self._prepare_input(x)  # [B, H*W, C]
        
        # Apply dropout for regularization
        x_seq = self.dropout(x_seq)
        
        # Process through Mamba - the key component
        x_seq = self.mamba(x_seq)
        
        # Restore spatial dimensions
        x_out = self._restore_spatial(x_seq, h, w)
        
        return x_out

class MambaBackbone(nn.Module):
    """Mamba-based backbone for video processing with efficient memory usage."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        d_state: int = 16,
        dropout: float = 0.1,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_channels = input_dim
        
        for dim in hidden_dims:
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ))
            in_channels = dim
        
        # Mamba blocks - the key innovation
        self.mamba_blocks = nn.ModuleList([
            VideoMambaBlock(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for dim in hidden_dims
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through backbone with Mamba processing."""
        features = []
        
        # Process through downsampling blocks
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            
            # Process through Mamba block - using checkpointing for memory efficiency if needed
            if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
                mamba_out = torch.utils.checkpoint.checkpoint(
                    self.mamba_blocks[i], 
                    x,
                    preserve_rng_state=False
                )
            else:
                mamba_out = self.mamba_blocks[i](x)
            
            features.append(mamba_out)
        
        return features

class FeatureFusion(nn.Module):
    """Feature fusion module to combine features from different scales."""
    
    def __init__(self, feature_dims: List[int], out_dim: int):
        super().__init__()
        
        # Projection layers to uniform dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, out_dim, kernel_size=1)
            for dim in feature_dims
        ])
        
        # Final fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Combine features from different scales."""
        # Highest resolution feature is the reference
        target_size = features[0].shape[-2:]
        
        # Project each feature to common dimension
        projected = []
        for i, feature in enumerate(features):
            x = self.projections[i](feature)
            
            # Upsample if needed
            if i > 0:
                x = F.interpolate(
                    x, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            projected.append(x)
        
        # Sum all projected features
        fused = sum(projected)
        
        # Apply final fusion
        return self.fusion(fused)

class BinaryVideoMambaSegmentation(nn.Module):
    """Binary video segmentation model with Mamba backbone."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract model configurations
        self.input_dim = config['input_dim']
        self.hidden_dims = config['hidden_dims']
        
        # Create Mamba backbone - the key innovation of your architecture
        self.backbone = MambaBackbone(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            d_state=config['d_state'],
            dropout=config.get('dropout', 0.1),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2)
        )
        
        # Feature fusion module
        self.feature_fusion = FeatureFusion(
            feature_dims=self.hidden_dims,
            out_dim=self.hidden_dims[0]  # Use first hidden dim
        )
        
        # Binary segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dims[0], 1, kernel_size=1)
        )
        
        # Temporal smoothing for video consistency
        self.temporal_smooth = nn.Conv3d(
            1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with frame-by-frame processing and temporal smoothing."""
        B, T, C, H, W = x.shape
        
        # Process each frame independently to avoid memory issues
        all_logits = []
        
        for t in range(T):
            # Get current frame
            frame = x[:, t]  # [B, C, H, W]
            
            # Process through backbone
            features = self.backbone(frame)
            
            # Fuse multi-scale features
            fused = self.feature_fusion(features)
            
            # Generate segmentation mask
            logits = self.seg_head(fused)
            all_logits.append(logits)
        
        # Stack results along temporal dimension
        stacked_logits = torch.stack(all_logits, dim=1)  # [B, T, 1, H, W]
        
        # Apply temporal smoothing
        smooth_logits = stacked_logits.permute(0, 2, 1, 3, 4)  # [B, 1, T, H, W]
        smooth_logits = self.temporal_smooth(smooth_logits)
        smooth_logits = smooth_logits.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
        
        return {
            'logits': smooth_logits,
            'pred_masks': torch.sigmoid(smooth_logits)
        }

def build_model(config: Dict) -> nn.Module:
    """Build binary video segmentation model with Mamba backbone."""
    if 'model' in config:
        config = config['model']
    
    return BinaryVideoMambaSegmentation(config)