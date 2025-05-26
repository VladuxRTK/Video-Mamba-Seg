import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

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
        
        # Initialize Mamba block
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Add normalization
        self.norm = None
    
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
        
        # Process through Mamba
        x_seq = self.mamba(x_seq)
        
        # Restore spatial dimensions
        x_out = self._restore_spatial(x_seq, h, w)
        
        # Add batch normalization
        if self.norm is None:
            self.norm = nn.BatchNorm2d(c).to(x.device)
        
        x_out = self.norm(x_out)
        
        return x_out

class MambaBackbone(nn.Module):
    """Simplified Mamba backbone for video processing."""
    
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
        
        # Mamba blocks
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
            
            # Process through Mamba block
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

class SimpleBinaryVideoMambaSegmentation(nn.Module):
    """Simplified binary video segmentation model - working version."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract model configurations
        self.input_dim = config['input_dim']
        self.hidden_dims = config['hidden_dims']
        
        # Create Mamba backbone
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
            out_dim=self.hidden_dims[0]
        )
        
        # Binary segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dims[0], 1, kernel_size=1)
        )
        
        # Simple temporal smoothing with 1D convolution
        self.temporal_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        print("Applied simplified model weight initialization")
    
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified forward pass without complex temporal processing."""
        B, T, C, H, W = x.shape
        print(f"Input shape: [B={B}, T={T}, C={C}, H={H}, W={W}]")
        
        # Process each frame independently
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
        print(f"Stacked logits shape: {stacked_logits.shape}")
        
        # Simple temporal smoothing if multiple frames
        if T > 1:
            # Reshape for 1D temporal convolution: [B*H*W, 1, T]
            B, T, C, H, W = stacked_logits.shape
            reshaped = stacked_logits.permute(0, 3, 4, 2, 1).reshape(B*H*W, C, T)
            
            # Apply temporal smoothing
            smoothed = self.temporal_smooth(reshaped)
            
            # Reshape back: [B*H*W, 1, T] -> [B, T, 1, H, W]
            smooth_logits = smoothed.reshape(B, H, W, C, T).permute(0, 4, 3, 1, 2)
        else:
            smooth_logits = stacked_logits
        
        print(f"Final logits shape: {smooth_logits.shape}")
        
        # Generate probabilities
        pred_probs = torch.sigmoid(smooth_logits)
        print(f"Prediction range: {pred_probs.min().item():.4f} to {pred_probs.max().item():.4f}")
        
        # Simple adaptive thresholding
        adaptive_masks = self._simple_adaptive_thresholding(pred_probs, B, T)
        
        return {
            'logits': smooth_logits,
            'pred_masks': pred_probs,
            'adaptive_masks': adaptive_masks
        }
    
    def _simple_adaptive_thresholding(self, pred_probs: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Simple adaptive thresholding without temporal memory."""
        expected_fg_ratio = 0.12
        adaptive_masks = []
        
        for b in range(B):
            batch_adaptive = []
            
            for t in range(T):
                # Get current frame prediction
                curr_pred = pred_probs[b, t, 0]  # [H, W]
                
                # Calculate adaptive threshold
                flat_pred = curr_pred.flatten()
                sorted_pred, _ = torch.sort(flat_pred, descending=True)
                threshold_idx = int(len(flat_pred) * expected_fg_ratio)
                
                if threshold_idx < len(sorted_pred):
                    threshold = sorted_pred[threshold_idx].item()
                else:
                    threshold = 0.5
                
                # Ensure reasonable bounds
                threshold = max(0.1, min(0.9, threshold))
                
                # Apply threshold
                adaptive_mask = (curr_pred > threshold).float()
                batch_adaptive.append(adaptive_mask)
                
                # Debug info
                pred_fg_ratio = adaptive_mask.mean().item()
                print(f"  Frame {t}: thresh={threshold:.3f}, FG ratio={pred_fg_ratio:.3f}")
            
            batch_masks = torch.stack(batch_adaptive, dim=0)
            adaptive_masks.append(batch_masks)
        
        # Stack and add channel dimension
        adaptive_masks = torch.stack(adaptive_masks, dim=0)
        adaptive_masks = adaptive_masks.unsqueeze(2)  # [B, T, 1, H, W]
        
        return adaptive_masks

def build_model(config: Dict) -> SimpleBinaryVideoMambaSegmentation:
    """Build simplified model."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
        
    model_config = config.get('model', config)
    required_params = ['input_dim', 'hidden_dims', 'd_state']
    missing_params = [param for param in required_params if param not in model_config]
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    return SimpleBinaryVideoMambaSegmentation(model_config)