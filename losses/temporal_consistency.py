import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class TemporalConsistencyLoss(nn.Module):
    """Enhanced temporal consistency loss for binary segmentation."""
    
    def __init__(self, consistency_weight: float = 1.0, smoothness_weight: float = 0.5):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        pred_masks: torch.Tensor,  # [B, T, 1, H, W]
        flows: Optional[torch.Tensor] = None  # [B, T-1, 2, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate temporal consistency loss for binary segmentation.
        
        Args:
            pred_masks: Predicted segmentation masks
            flows: Optional optical flow between consecutive frames
            
        Returns:
            Dictionary containing temporal losses
        """
        if pred_masks.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got shape {pred_masks.shape}")
            
        B, T, C, H, W = pred_masks.shape
        losses = {}
        
        if T <= 1:
            # No temporal loss for single frame
            return {'temporal_loss': torch.tensor(0.0, device=pred_masks.device)}
        
        # Basic temporal smoothness - penalize large changes between frames
        temporal_diff = pred_masks[:, 1:] - pred_masks[:, :-1]  # [B, T-1, C, H, W]
        
        # L2 smoothness loss
        temporal_loss = F.mse_loss(temporal_diff, torch.zeros_like(temporal_diff))
        losses['temporal_loss'] = temporal_loss * self.consistency_weight
        
        # Additional smoothness constraint - favor gradual changes
        if T > 2:
            # Second-order smoothness (acceleration penalty)
            second_diff = temporal_diff[:, 1:] - temporal_diff[:, :-1]  # [B, T-2, C, H, W]
            smoothness_loss = F.mse_loss(second_diff, torch.zeros_like(second_diff))
            losses['smoothness_loss'] = smoothness_loss * self.smoothness_weight
        
        return losses