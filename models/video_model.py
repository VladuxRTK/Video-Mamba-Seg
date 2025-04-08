import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F

from .backbone import BackboneEncoder
from .video_instance_decoder import VideoInstanceDecoder
from .temporal_components import EnhancedTemporalSmoothingModule

# In models/video_model.py, modify the VideoMambaSegmentation class

class VideoMambaSegmentation(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.feature_dim = config['hidden_dims'][-1]
        # Remove num_instances parameter since we're doing binary segmentation now
        
        # Initialize backbone (keep this as is)
        self.backbone = BackboneEncoder(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            d_state=config['d_state'],
            temporal_window=config['temporal_window'],
            dropout=config.get('dropout', 0.1),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2)
        )
        
        # Replace instance decoder with binary segmentation decoder
        self.seg_head = BinarySegmentationHead(
            in_channels=config['hidden_dims'],
            hidden_dim=256
        )
        
        # Temporal smoothing can be reused but simplified
        self.temporal_smooth = EnhancedTemporalSmoothingModule(
            channels=1,  # Now only one channel for binary segmentation
            temporal_kernel=3
        )
    
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with proper dimension handling for video data.
        
        Args:
            x: Input tensor [B, T, C, H, W] representing a video sequence
                
        Returns:
            Dictionary containing model outputs
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Process through binary segmentation head
        logits = self.seg_head(backbone_features)
        
        # Reshape to add temporal dimension [B, T, 1, H, W]
        logits = logits.view(B, T, 1, logits.shape[2], logits.shape[3])
        
        # Apply temporal smoothing
        smoothed_logits = self.temporal_smooth(logits)
        
        # Return dictionary with both logits for loss computation and probabilities for visualization
        return {
            'pred_masks': torch.sigmoid(smoothed_logits),  # For visualization and evaluation
            'logits': smoothed_logits  # For loss computation
        }

# Add this to models/video_model.py or create a new file

class BinarySegmentationHead(nn.Module):
    def __init__(self, in_channels: List[int], hidden_dim: int = 256):
        super().__init__()
        
        # Feature projections from each backbone level
        self.projections = nn.ModuleList([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
            for in_dim in in_channels
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(in_channels), hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction layer (just 1 channel for binary segmentation)
        self.predictor = nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Project and resize all feature levels to the highest resolution
        proj_features = []
        
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            projected = proj(feat)
            
            # Resize to match the size of the highest resolution feature
            if i > 0:  # Skip the first (highest resolution) feature
                target_size = features[0].shape[-2:]
                projected = F.interpolate(projected, size=target_size, mode='bilinear', align_corners=False)
            
            proj_features.append(projected)
        
        # Concatenate all feature levels
        fused_features = torch.cat(proj_features, dim=1)
        
        # Apply feature fusion
        fused = self.fusion(fused_features)
        
        # Final prediction
        logits = self.predictor(fused)
        
        return logits

def build_model(config: Dict) -> VideoMambaSegmentation:
    """
    Builds and initializes the model from config.
    Handles both nested and flat configuration formats.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
        
    # Handle nested configuration
    model_config = config.get('model', config)
    
    # Verify required parameters
    required_params = ['input_dim', 'hidden_dims', 'd_state']
    missing_params = [param for param in required_params if param not in model_config]
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Create model
    return VideoMambaSegmentation(model_config)