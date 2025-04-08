import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

from .position_encoding import LeveledPositionEmbeddingSine
from .temporal_components import InstanceMotionModule,InstanceTemporalAttention

class InstanceMemory(nn.Module):
    """
    Memory module that tracks instance features across frames.
    Provides temporal consistency for instance identity.
    """
    def __init__(self, feature_dim: int, num_instances: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_instances = num_instances
        
        # GRU cell for updating instance memory
        self.memory_update = nn.GRUCell(
            input_size=feature_dim,
            hidden_size=feature_dim
        )
        
        # Instance association score
        self.association_score = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, instance_features: torch.Tensor, time_idx: int) -> torch.Tensor:
        """
        Update instance memory with new features.
        
        Args:
            instance_features: Features of current instances [B, N, C]
            time_idx: Current time index
            
        Returns:
            Updated instance memory [B, N, C]
        """
        B, N, C = instance_features.shape
        
        # First frame - initialize memory
        if time_idx == 0:
            return instance_features
        
        # Flatten batch and instance dimensions
        flat_features = instance_features.view(B * N, C)
        
        # Update memory
        updated_memory = self.memory_update(
            flat_features,
            flat_features  # Use current features as hidden state for simplicity
        )
        
        # Reshape back to [B, N, C]
        updated_memory = updated_memory.view(B, N, C)
        
        return updated_memory


class VideoInstanceDecoder(nn.Module):
    """
    Video instance decoder that maintains instance identity across frames.
    Uses query-based attention with explicit temporal handling.
    """
    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 256,
        num_instances: int = 16,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Feature projections for each level
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, kernel_size=1)
            for dim in in_channels
        ])
        
        # Position encodings that match the INPUT feature dimensions
        self.position_encodings = nn.ModuleList([
            LeveledPositionEmbeddingSine(dim)  # Use original input channel dimensions
            for dim in in_channels
        ])
        
        # Instance query embeddings (learnable)
        self.instance_queries = nn.Parameter(
            torch.randn(num_instances, hidden_dim)
        )
        
        # Add InstanceMotionModule for motion modeling
        self.instance_motion = InstanceMotionModule(
            feature_dim=hidden_dim,
            num_instances=num_instances
        )
        
        # Add InstanceTemporalAttention for temporal feature processing
        self.temporal_attention = InstanceTemporalAttention(
            feature_dim=hidden_dim,
            num_instances=num_instances,
            num_heads=num_heads
        )
        
        # Temporal instance memory to track instances across frames
        self.instance_memory = InstanceMemory(
            feature_dim=hidden_dim,
            num_instances=num_instances
        )
        
        # Cross-frame attention to link instances between frames
        self.query_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Mask prediction head
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
    
    def forward(
        self, 
        features: List[List[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of frames with instance tracking.
        
        Args:
            features: List of feature lists for each frame
                     [frames][levels][B, C, H, W]
        
        Returns:
            Dictionary with pred_masks of shape [B, T, N, H, W]
        """
        T = len(features)  # Number of frames
        B = features[0][0].shape[0]  # Batch size
        
        # 1. Apply position encoding and project features for each frame
        projected_features = []
        
        for t in range(T):
            frame_features = features[t]
            # First apply position encoding to input features
            positioned_features = []
            for i, (feat, pos_enc) in enumerate(zip(frame_features, self.position_encodings)):
                pos = pos_enc(feat)  # Generate position encoding
                feat_with_pos = feat + pos  # Add position encoding to features
                positioned_features.append(feat_with_pos)
            
            # Then project to hidden dimension
            projected = [
                proj(feat) for feat, proj in zip(positioned_features, self.feature_projections)
            ]
            projected_features.append(projected)
        
        # 2. Process through motion module and temporal attention if T > 1
        if T > 1:
            # Extract highest resolution features for motion processing
            motion_input = torch.stack([
                projected_features[t][0] for t in range(T)
            ], dim=1)  # [B, T, C, H, W]
            
            # Apply motion module
            motion_features, motion_field = self.instance_motion(motion_input)
            
            # Apply temporal attention
            attended_features = self.temporal_attention(motion_features, motion_field)
            
            # Update the projected features with motion-aware information
            for t in range(T):
                projected_features[t][0] = attended_features[:, t]
        else:
            motion_field = None
            
        # 3. Process frames sequentially to maintain instance tracking
        all_masks = []
        prev_instances = None
        
        for t in range(T):
            # Use highest resolution features
            feat = projected_features[t][0]  # [B, C, H, W]
            C, H, W = feat.shape[1:]
            
            # Initialize instance queries
            if prev_instances is None:
                # First frame - use learnable queries
                instance_queries = self.instance_queries.unsqueeze(0).expand(B, -1, -1)
            else:
                # Use previous frame instances with attention
                instance_queries, _ = self.query_attention(
                    self.instance_queries.unsqueeze(0).expand(B, -1, -1),
                    prev_instances,
                    prev_instances
                )
            
            # Generate instance features through attention
            flat_features = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            
            # Each instance generates its mask
            instance_masks = []
            instance_features = []
            
            for i in range(instance_queries.size(1)):
                query = instance_queries[:, i:i+1, :]  # [B, 1, C]
                
                # Compute attention between query and features
                attn = torch.bmm(query, flat_features.transpose(1, 2))  # [B, 1, H*W]
                attn = F.softmax(attn / math.sqrt(C), dim=2)
                
                # Generate instance-specific features
                inst_feat = torch.bmm(attn, flat_features)  # [B, 1, C]
                instance_features.append(inst_feat)
                
                # Generate mask for this instance
                mask_feat = inst_feat.view(B, C, 1, 1).expand(-1, -1, H, W)
                mask = self.mask_predictor(mask_feat)
                instance_masks.append(mask)
            
            # Stack all instance masks
            frame_masks = torch.cat(instance_masks, dim=1)  # [B, N, H, W]
            all_masks.append(frame_masks)
            
            # Stack instance features
            frame_features = torch.cat(instance_features, dim=1)  # [B, N, C]
            
            # Update instance memory for next frame
            prev_instances = self.instance_memory(frame_features, t)
        
        # 4. Stack masks from all frames [B, T, N, H, W]
        pred_masks = torch.stack(all_masks, dim=1)
        
        # 5. Return results with motion field if available
        if motion_field is not None:
            return {
                "pred_masks": pred_masks,
                "motion_field": motion_field
            }
        else:
            return {"pred_masks": pred_masks}