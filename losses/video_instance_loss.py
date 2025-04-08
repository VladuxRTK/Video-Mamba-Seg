import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss."""
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss."""
        # Apply sigmoid if needed
        if not (0 <= pred.min() <= 1 and 0 <= pred.max() <= 1):
            pred = torch.sigmoid(pred)
        
        # For numerical stability
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        
        # Compute focal loss
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # Compute binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Combine
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Loss to encourage temporal consistency between consecutive frames.
    """
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        pred_masks: torch.Tensor,  # [B, T, N, H, W]
        flows: Optional[torch.Tensor] = None  # Optional flow fields
    ) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Simple temporal difference penalty
        if pred_masks.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,T,N,H,W], got shape {pred_masks.shape}")
            
        B, T, N, H, W = pred_masks.shape
        
        # No temporal loss for single frame
        if T <= 1:
            return torch.tensor(0.0, device=pred_masks.device)
        
        # Calculate difference between consecutive frames
        temporal_diff = pred_masks[:, 1:] - pred_masks[:, :-1]  # [B, T-1, N, H, W]
        
        # Compute L2 loss
        temporal_loss = F.mse_loss(temporal_diff, torch.zeros_like(temporal_diff))
        
        return temporal_loss * self.consistency_weight


class VideoInstanceSegmentationLoss(nn.Module):
    """
    Combined loss for video instance segmentation with temporal consistency.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        temporal_weight: float = 0.5
    ):
        super().__init__()
        
        # Segmentation losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
        # Temporal consistency loss
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
        
        # Loss weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        flows: Optional[torch.Tensor] = None  # Optional flow fields
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for video instance segmentation.
        
        Args:
            outputs: Dictionary containing 'pred_masks' [B, T, N, H, W]
            targets: Dictionary containing 'masks' [B, T, H, W]
            flows: Optional flow fields between frames
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Get predictions and ground truth
        pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
        gt_masks = targets['masks']         # [B, T, H, W]
        
        B, T, N, H, W = pred_masks.shape
        
        # Initialize losses
        dice_loss = 0
        focal_loss = 0
        
        # For each instance in prediction, find best matching ground truth
        instance_ids = torch.unique(gt_masks)[1:]  # Skip background
        
        # Simple matching strategy: for each ground truth instance, compute loss
        # against best matching predicted instance
        for instance_id in instance_ids:
            # Create binary mask for this instance
            gt_instance = (gt_masks == instance_id).float()  # [B, T, H, W]
            
            # Find best matching predicted instance
            best_iou = -1
            best_idx = -1
            
            for n in range(N):
                pred_instance = pred_masks[:, :, n]  # [B, T, H, W]
                
                # Compute IoU
                intersection = (pred_instance > 0.5).float() * gt_instance
                union = (pred_instance > 0.5).float() + gt_instance - intersection
                
                iou = (intersection.sum() + 1e-6) / (union.sum() + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = n
            
            if best_idx >= 0:
                # Compute losses for best matching instance
                pred_instance = pred_masks[:, :, best_idx]  # [B, T, H, W]
                
                # Dice loss
                dice = self.dice_loss(pred_instance, gt_instance)
                dice_loss += dice
                
                # Focal loss
                focal = self.focal_loss(pred_instance, gt_instance)
                focal_loss += focal
        
        # Normalize by number of instances
        num_instances = max(1, len(instance_ids))
        dice_loss /= num_instances
        focal_loss /= num_instances
        
        # Compute temporal consistency loss
        temp_loss = self.temporal_loss(pred_masks, flows)
        
        # Weighted combination
        total_loss = (
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss +
            temp_loss
        )
        
        return {
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'temporal_loss': temp_loss,
            'total_loss': total_loss
        }