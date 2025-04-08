import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,Tuple

# In losses/segmentation.py

# In losses/segmentation.py, modify the BinarySegmentationLoss class
class BinarySegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 1.5, boundary_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()  # New boundary loss
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract predictions and targets
        if 'logits' in outputs:
            logits = outputs['logits']  # [B, T, 1, H, W]
        else:
            # Ensure logits is defined by using pred_masks
            logits = outputs['pred_masks']  # May already be probabilities
            
        # Now logits is defined in all code paths
        ce_loss = self.ce_loss(logits.squeeze(1), binary_masks) * self.ce_weight
        dice_loss = self.dice_loss(pred_probs.squeeze(1), binary_masks) * self.dice_weight
        
        # Add boundary loss
        boundary_loss = self.boundary_loss(pred_probs.squeeze(1), binary_masks) * self.boundary_weight
        
        # Compute total loss
        total_loss = ce_loss + dice_loss + boundary_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss
        }

# Add a new boundary loss class
class BoundaryLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # Create boundary maps
        if self.sobel_x.device != predictions.device:
            self.sobel_x = self.sobel_x.to(predictions.device)
            self.sobel_y = self.sobel_y.to(predictions.device)
        
        # Get edges from predictions
        pred_boundaries = self._get_boundaries(predictions.unsqueeze(1))
        
        # Get edges from targets
        target_boundaries = self._get_boundaries(targets.unsqueeze(1).float())
        
        # Calculate boundary IoU loss
        intersection = (pred_boundaries * target_boundaries).sum(dim=[1, 2, 3])
        union = pred_boundaries.sum(dim=[1, 2, 3]) + target_boundaries.sum(dim=[1, 2, 3]) - intersection
        boundary_iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return (1 - boundary_iou).mean()
    
    def _get_boundaries(self, tensor):
        # Apply Sobel filters for edge detection
        grad_x = F.conv2d(tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor, self.sobel_y, padding=1)
        
        # Calculate gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and threshold
        grad_mag = grad_mag / grad_mag.max()
        return grad_mag
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss between predictions and targets.
        
        Args:
            predictions: Predicted probabilities [B, H, W]
            targets: Binary target masks [B, H, W]
            
        Returns:
            Dice loss
        """
        batch_size = predictions.size(0)
        
        # Flatten predictions and targets
        pred_flat = predictions.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Compute intersection and union
        intersection = (pred_flat * targets_flat).sum(1)
        union = pred_flat.sum(1) + targets_flat.sum(1)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice.mean()