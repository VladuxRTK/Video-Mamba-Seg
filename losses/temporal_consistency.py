import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class TemporalConsistencyLoss(nn.Module):
    """Loss to enforce temporal consistency between frames."""
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        pred_masks: torch.Tensor,  # [B, T, C, H, W]
        flows: Optional[torch.Tensor] = None  # [B, T-1, 2, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate temporal consistency loss between consecutive frames.
        
        Args:
            pred_masks: Predicted segmentation masks
            flows: Optional optical flow between consecutive frames
            
        Returns:
            Dictionary containing:
                - 'temporal_loss': Main temporal consistency loss
                - 'smoothness_loss': Optional flow smoothness loss if flows provided
        """
        if pred_masks.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got shape {pred_masks.shape}")
            
        B, T, C, H, W = pred_masks.shape
        losses = {}
        
        # Basic temporal consistency - difference between consecutive frames
        temporal_diff = pred_masks[:, 1:] - pred_masks[:, :-1]  # [B, T-1, C, H, W]
        temporal_loss = F.mse_loss(temporal_diff, torch.zeros_like(temporal_diff))
        losses['temporal_loss'] = temporal_loss * self.consistency_weight
        
        # If flows provided, use them for warped consistency
        if flows is not None:
            warped_masks = []
            for t in range(T-1):
                curr_flow = flows[:, t]  # [B, 2, H, W]
                next_mask = pred_masks[:, t+1]  # [B, C, H, W]
                
                # Create sampling grid from flow
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=flows.device),
                    torch.arange(W, device=flows.device),
                    indexing='ij'
                )
                grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
                grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
                
                # Add flow to grid
                flow_grid = grid + curr_flow
                
                # Normalize grid coordinates to [-1, 1]
                flow_grid[:, 0] = 2.0 * flow_grid[:, 0] / (W - 1) - 1.0
                flow_grid[:, 1] = 2.0 * flow_grid[:, 1] / (H - 1) - 1.0
                
                # Reshape grid for grid_sample
                flow_grid = flow_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
                
                # Warp masks using flow
                warped_mask = F.grid_sample(
                    next_mask,
                    flow_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )
                warped_masks.append(warped_mask)
            
            warped_masks = torch.stack(warped_masks, dim=1)  # [B, T-1, C, H, W]
            
            # Calculate warped consistency loss
            warped_loss = F.mse_loss(
                pred_masks[:, :-1],
                warped_masks.detach()
            )
            losses['warped_loss'] = warped_loss * self.consistency_weight
            
            # Optional flow smoothness loss
            if self.training:
                flow_gradients_x = flows[:, :, :, :, 1:] - flows[:, :, :, :, :-1]
                flow_gradients_y = flows[:, :, :, 1:, :] - flows[:, :, :, :-1, :]
                smoothness_loss = (flow_gradients_x.abs().mean() + 
                                 flow_gradients_y.abs().mean())
                losses['smoothness_loss'] = smoothness_loss * 0.1
        
        return losses