import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class DimensionAdapter(nn.Module):
    """
    Handles dimension adaptation between different parts of the model.
    This module ensures smooth transitions between different feature spaces,
    acting like a universal translator between feature dimensions.
    """
    def __init__(self, feature_dim: int, mask_dim: int):
        super().__init__()
        
        # Feature space transformations
        self.feature_to_mask = nn.Sequential(
            nn.Linear(feature_dim, mask_dim),
            nn.LayerNorm(mask_dim),
            nn.ReLU(inplace=True)
        )
        
        self.mask_to_feature = nn.Sequential(
            nn.Linear(mask_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def adapt_features(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Adapts feature dimensions based on target size.
        Args:
            x: Input tensor
            target_dim: Desired output dimension
        """
        curr_dim = x.shape[-1]
        if curr_dim == target_dim:
            return x
        elif target_dim == self.mask_to_feature[-2].normalized_shape[0]:
            return self.mask_to_feature(x)
        else:
            return self.feature_to_mask(x)

class FlexibleTemporalAttention(nn.Module):
    """
    Temporal attention module that automatically handles dimension matching
    between features and queries.
    """
    def __init__(self, feature_dim: int, mask_dim: int, num_instances: int):
        super().__init__()
        
        self.dim_adapter = DimensionAdapter(feature_dim, mask_dim)
        
        # Temporal processing
        self.temporal_conv = nn.Conv3d(
            feature_dim, feature_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=feature_dim // 16 if feature_dim >= 16 else 1
        )
        
        # Cross-attention for instance features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim * 2, feature_dim, 1),
            nn.GroupNorm(min(8, feature_dim), feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, feature_dim, 32, 1, 1))
    
    def forward(self, features: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Process features with automatic dimension handling.
        Args:
            features: [B, C, T, H, W] backbone features
            queries: [B, N, D] instance queries
        """
        B, C, T, H, W = features.shape
        
        # Adapt query dimensions to match feature space
        adapted_queries = self.dim_adapter.adapt_features(queries, C)
        
        # Reshape features for attention
        feat_flat = features.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        feat_flat = feat_flat.reshape(B, T*H*W, C)   # [B, THW, C]
        
        # Apply cross-attention
        attended_feats, _ = self.cross_attention(
            feat_flat, adapted_queries, adapted_queries
        )
        
        # Reshape back to feature format
        attended_feats = attended_feats.view(B, T, H, W, C)
        attended_feats = attended_feats.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        
        # Apply temporal processing
        features = features + self.pos_encoding[:, :, :T]
        temporal_feats = self.temporal_conv(features)
        
        # Combine features
        combined = torch.cat([attended_feats, temporal_feats], dim=1)
        enhanced = self.feature_refine(combined)
        
        return enhanced

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class InstanceMotionModule(nn.Module):
    """
    Module responsible for tracking instance-specific motion between frames.
    This helps maintain instance identity by understanding how objects move
    through the video sequence.
    """
    def __init__(self, feature_dim: int, num_instances: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_instances = num_instances
        
        # Motion estimation network
        self.motion_estimator = nn.Sequential(
            # First layer processes features
            nn.Conv3d(feature_dim, feature_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True),
            # Second layer refines motion features
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True),
            # Final layer predicts motion field
            nn.Conv3d(feature_dim, 2, kernel_size=1)  # 2 channels for x,y motion
        )
        
        # Feature refinement with motion awareness
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim + 2, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process features to estimate motion and enhance feature representation.
        
        Args:
            features: Input features [B, T, C, H, W]
            
        Returns:
            Tuple containing:
            - Enhanced features incorporating motion information
            - Estimated motion field between consecutive frames
        """
        B, T, C, H, W = features.shape
        
        # Reshape for 3D convolution
        x = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Estimate motion
        motion = self.motion_estimator(x)  # [B, 2, T, H, W]
        
        # Get motion field between consecutive frames
        motion_field = motion.permute(0, 2, 1, 3, 4)  # [B, T, 2, H, W]
        
        # Combine features with motion information
        motion_features = torch.cat([x, motion], dim=1)
        enhanced = self.feature_refine(motion_features)
        
        # Return to original format
        enhanced = enhanced.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        
        return enhanced, motion_field

class InstanceTemporalAttention(nn.Module):
    """
    Enhanced temporal attention module that maintains instance identity across video frames
    while properly handling the output shapes to ensure compatibility with the spatial dimensions.
    """
    def __init__(
        self,
        feature_dim: int,
        num_instances: int,
        num_heads: int = 8
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_instances = num_instances
        self.num_heads = num_heads
        
        # Initialize learnable instance query embeddings
        self.instance_queries = nn.Parameter(
            torch.randn(num_instances, feature_dim)
        )
        
        # Multi-head attention for instance-feature interactions
        self.instance_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Temporal processing with 3D convolution
        self.temporal_conv = nn.Conv3d(
            feature_dim,
            feature_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=feature_dim // 16 if feature_dim >= 16 else 1
        )
        
        # Feature refinement combining temporal and instance information
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim * 2, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Position encoding for temporal awareness
        self.pos_encoding = nn.Parameter(
            torch.randn(1, feature_dim, 32, 1, 1)
        )
        
        # Instance feature projection to spatial domain
        self.instance_to_spatial = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        motion_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process features with both temporal attention and instance tracking,
        with corrected reshaping to maintain spatial dimensions.
        
        Args:
            features: Input features [B, T, C, H, W]
            motion_field: Optional motion information [B, T-1, 2, H, W]
            
        Returns:
            Enhanced features with temporal and instance awareness [B, T, C, H, W]
        """
        B, T, C, H, W = features.shape
        
        # Print shapes for debugging
        # print(f"\nProcessing in temporal attention:")
        # print(f"Input features shape: {features.shape}")
        
        # 1. General temporal processing with 3D convolution
        temporal_feats = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        temporal_feats = temporal_feats + self.pos_encoding[:, :, :T]
        temporal_feats = self.temporal_conv(temporal_feats)
        # print(f"After temporal conv: {temporal_feats.shape}")
        
        # 2. Instance-specific processing - first get feature vectors
        # Reshape features for attention by flattening spatial dimensions
        feat_flat = features.reshape(B * T, H * W, C)
        # print(f"Flattened features: {feat_flat.shape}")
        
        # Reshape instance queries for attention
        queries = self.instance_queries.unsqueeze(0).expand(B * T, -1, -1)
        # print(f"Instance queries shape: {queries.shape}")
        
        # Apply instance-specific attention
        # Instance queries attend to spatial features
        instance_feats, _ = self.instance_attention(
            queries,            # Queries from instance embeddings [B*T, N, C]
            feat_flat,          # Keys from flattened features [B*T, H*W, C]
            feat_flat           # Values from flattened features [B*T, H*W, C]
        )
        # print(f"After attention: {instance_feats.shape}")  # [B*T, N, C]
        
        # 3. Project instance features back to spatial domain
        # We use a different approach that doesn't require reshape to full spatial size
        
        # First get instance features for each position
        # Create attention map from instances to spatial locations
        instance_attn = torch.bmm(
            feat_flat,                          # [B*T, H*W, C]
            instance_feats.transpose(1, 2)      # [B*T, C, N]
        )  # [B*T, H*W, N]
        
        # Normalize attention weights
        instance_attn = F.softmax(instance_attn, dim=2)
        
        # Get weighted instance features for each spatial location
        spatial_instance_feats = torch.bmm(
            instance_attn,                      # [B*T, H*W, N]
            instance_feats                      # [B*T, N, C]
        )  # [B*T, H*W, C]
        
        # Reshape back to spatial format
        spatial_instance_feats = spatial_instance_feats.reshape(B, T, H, W, C)
        spatial_instance_feats = spatial_instance_feats.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        
        # Now reshape to match the temporal processing format
        instance_spatial = spatial_instance_feats.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        # print(f"Instance features mapped to spatial: {instance_spatial.shape}")
        
        # 4. Combine temporal and instance features
        combined = torch.cat([temporal_feats, instance_spatial], dim=1)
        enhanced = self.feature_refine(combined)
        # print(f"Final enhanced features: {enhanced.shape}")
        
        # Return to original format [B, T, C, H, W]
        output = enhanced.permute(0, 2, 1, 3, 4)
        return output.contiguous()

# Reuse your existing EnhancedTemporalSmoothingModule but modify it for binary segmentation
# In models/temporal_components.py

class EnhancedTemporalSmoothingModule(nn.Module):
    """
    Applies temporal smoothing while maintaining consistent segmentation.
    Modified to work with binary segmentation (1 channel) instead of instance segmentation.
    """
    def __init__(
        self,
        channels: int = 1,  # Now default to 1 for binary segmentation
        temporal_kernel: int = 3
    ):
        super().__init__()
        
        # Calculate appropriate number of groups
        num_groups = 1  # For binary segmentation, just use 1 group
        
        # Temporal smoothing
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel//2, 0, 0),
            groups=num_groups
        )
        
        # Feature refinement
        self.segment_refine = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing to segmentation features.
        
        Args:
            features: Segmentation features [B, T, C, H, W] where C is now 1
            
        Returns:
            Temporally smoothed features with same shape
        """
        # Reshape for temporal processing
        B, T, C, H, W = features.shape
        features = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Apply smoothing
        identity = features
        smoothed = self.temporal_conv(features)
        enhanced = self.segment_refine(smoothed + identity)
        
        # Return to original shape
        return enhanced.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]