import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba as MambaBlock
from collections import deque
from typing import Optional, Tuple, List

class CNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Conv2d(curr_dim, hidden_dim, 3, padding=1),
                # Changed LayerNorm to BatchNorm2d for 4D tensors
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
            curr_dim = hidden_dim
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            features.append(current)
            
        return features

class VideoMambaBlock(nn.Module):
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
            expand = expand
        )
        
        self.state_reset_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.temporal_state = None
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape spatial dimensions into sequence length
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
    
    def _restore_spatial(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # Restore spatial dimensions
        b, hw, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x
        
    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False,
        motion_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not x.is_cuda:
            x = x.cuda()
        b, c, h, w = x.shape
        

        # Prepare input for Mamba
        x_seq = self._prepare_input(x)  # [B, H*W, C]
        
        # Apply dropout
        x_seq = self.dropout(x_seq)
        
        # Process through Mamba
        x_seq = self.mamba(x_seq)
        
        # Restore spatial dimensions
        x = self._restore_spatial(x_seq, h, w)
        
        # For now, just return processed tensor and None for state
        return x, None

class TemporalFeatureBank(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        window_size: int = 5,
        confidence_threshold: float = 0.7
    ):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.confidence_threshold = confidence_threshold
        self.features = deque(maxlen=window_size)
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            # Changed LayerNorm to BatchNorm2d
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def update(
        self,
        current_features: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> None:
        mask = confidence_scores > self.confidence_threshold
        self.features.append((current_features.detach(), mask))
        
    def get_temporal_context(self, current_features: torch.Tensor) -> torch.Tensor:
        if not self.features:
            return current_features
            
        # Get current feature dimensions
        B, C, H, W = current_features.shape
        
        temporal_features = []
        for hist_features, mask in self.features:
            # Skip features with wrong batch size
            if hist_features.shape[0] != B:
                continue
                
            # Resize historical features to match current feature size if needed
            if hist_features.shape[2:] != current_features.shape[2:]:
                hist_features = F.interpolate(
                    hist_features, 
                    size=(H, W),
                    mode='bilinear', 
                    align_corners=False
                )
                # Also resize mask
                if mask.shape[2:] != current_features.shape[2:]:
                    mask_float = mask.float()
                    mask_float = F.interpolate(
                        mask_float, 
                        size=(H, W),
                        mode='nearest'
                    )
                    mask = mask_float > 0.5
                    
            temporal_features.append(hist_features * mask)
            
        # Safely compute mean with proper reshaping
        if temporal_features:
            temporal_context = torch.cat([
                current_features,
                torch.mean(torch.stack(temporal_features), dim=0)
            ], dim=1)
        else:
            # If no temporal features yet, duplicate current features
            temporal_context = torch.cat([current_features, current_features], dim=1)
        
        return self.feature_fusion(temporal_context)

class BackboneEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        d_state: int = 16,
        temporal_window: int = 5,
        dropout: float = 0.1,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        self.cnn_backbone = CNNBackbone(input_dim, hidden_dims)
        
        self.mamba_blocks = nn.ModuleList([
            VideoMambaBlock(
                d_model=dim,
                d_state=d_state,
                d_conv = d_conv,
                expand = expand,
                dropout=dropout
            )
            for dim in hidden_dims
        ])
        
        self.temporal_banks = nn.ModuleList([
            TemporalFeatureBank(
                feature_dim=dim,
                window_size=temporal_window
            )
            for dim in hidden_dims
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        reset_states: bool = False,
        motion_info: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)
        
        # Process through Mamba blocks and Temporal Feature Banks
        enhanced_features = []
        for feat, mamba, bank in zip(cnn_features, self.mamba_blocks, self.temporal_banks):
            # Mamba processing
            mamba_features, _ = mamba(feat, reset_states, motion_info)
            
            # Temporal bank processing
            confidence = torch.norm(mamba_features, dim=1, keepdim=True)
            bank.update(mamba_features, confidence)
            temporal_features = bank.get_temporal_context(mamba_features)
            
            enhanced_features.append(temporal_features)
            
        return enhanced_features