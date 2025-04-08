import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .backbone import BackboneEncoder
from .decoder import MambaMask2FormerDecoder
from .temporal_components import (
    InstanceMotionModule,
    InstanceTemporalAttention,
    EnhancedTemporalSmoothingModule
)

class VideoMambaSegmentation(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.feature_dim = config['hidden_dims'][-1]
        self.mask_dim = config['mask2former']['mask_dim']
        self.num_instances = config['mask2former']['num_queries']
        
        print(f"\nInitializing complete video instance segmentation model:")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Mask dimension: {self.mask_dim}")
        print(f"- Number of instances: {self.num_instances}")
        
        # Initialize backbone
        self.backbone = BackboneEncoder(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            d_state=config['d_state'],
            temporal_window=config['temporal_window'],
            dropout=config.get('dropout', 0.1)
        )
        
        # Initialize instance-specific temporal components
        self.instance_motion = InstanceMotionModule(
            feature_dim=self.feature_dim,
            num_instances=self.num_instances
        )
        
        self.temporal_attention = InstanceTemporalAttention(
            feature_dim=self.feature_dim,
            num_instances=self.num_instances,
            num_heads=8
        )
        
        # Initialize mask projection
        self.mask_projection = nn.Sequential(
            nn.Conv2d(config['input_dim'], self.mask_dim // 2, 3, padding=1),
            nn.BatchNorm2d(self.mask_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mask_dim // 2, self.mask_dim, 1),
            nn.BatchNorm2d(self.mask_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize decoder with proper mask dimension
        self.decoder = MambaMask2FormerDecoder(
            in_channels=config['hidden_dims'],
            mask2former_config=config['mask2former'],
            num_classes=1  # Binary segmentation per instance
        )
        
        # Initialize temporal smoothing
        self.temporal_smooth = EnhancedTemporalSmoothingModule(
            channels=self.num_instances,
            temporal_kernel=3
        )
    
    def _generate_mask_features(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Generate mask features with temporal awareness."""
        # Project to mask dimension
        mask_features = self.mask_projection(x)  # [B*T, mask_dim, H, W]
        _, C, H, W = mask_features.shape
        return mask_features.view(B, T, C, H, W)
    
    def _process_temporal_features(
        self,
        features: torch.Tensor,
        B: int,
        T: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process features through temporal components."""
        C, H, W = features.shape[1:]
        features = features.view(B, T, C, H, W)
        
        # Track instance motion
        motion_features, motion_field = self.instance_motion(features)
        
        # Apply instance-specific temporal attention
        temporal_features = self.temporal_attention(
            motion_features,
            motion_field=motion_field
        )
        
        # Reshape back to batch format
        processed_features = temporal_features.reshape(B * T, C, H, W)
        return processed_features, motion_field
    
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
        
        # Generate mask features
        mask_features = self._generate_mask_features(x, B, T)
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Process through temporal components
        enhanced_features, motion_field = self._process_temporal_features(
            backbone_features[-1], B, T
        )
        backbone_features[-1] = enhanced_features
        
        # Process through decoder - no need to handle frames separately as it's done inside the decoder
        outputs = self.decoder(
            features=backbone_features,
            mask_features=mask_features
        )
        
        # Add motion field to outputs if available
        if motion_field is not None:
            outputs['motion_field'] = motion_field
        
        return outputs
    
    
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
    required_params = ['input_dim', 'hidden_dims', 'mask2former']
    missing_params = [param for param in required_params if param not in model_config]
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Create model
    return VideoMambaSegmentation(config)