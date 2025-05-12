import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

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
        """
        Compute combined loss for binary segmentation.
        
        Args:
            outputs: Dictionary containing 'logits' or 'pred_masks'
            targets: Dictionary containing 'masks'
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Extract predictions
        if 'logits' in outputs:
            logits = outputs['logits']  # Could be [B, T, 1, H, W]
        else:
            # Use pred_masks - may need to apply inverse sigmoid if they're probabilities
            pred_masks = outputs['pred_masks']
            # Check if they're probabilities (between 0 and 1)
            if pred_masks.max() <= 1.0 and pred_masks.min() >= 0.0:
                # Apply logit transform to convert back to logits
                eps = 1e-6  # Small value to prevent log(0) or log(1)
                logits = torch.log(pred_masks / (1 - pred_masks + eps) + eps)
            else:
                logits = pred_masks
        
        # Extract ground truth masks
        masks = targets['masks']  # Could be [B, T, H, W]
        
        # Handle batch+temporal dimensions
        if logits.dim() == 5 and masks.dim() == 4:
            # The shapes are [B, T, C, H, W] and [B, T, H, W]
            # We need to reshape to [B*T, C, H, W] and [B*T, H, W]
            B, T = masks.shape[:2]
            H, W = masks.shape[2:]
            
            # Reshape logits: [B, T, C, H, W] -> [B*T, C, H, W]
            logits = logits.reshape(B*T, logits.shape[2], H, W)
            
            # Reshape masks: [B, T, H, W] -> [B*T, H, W]
            masks = masks.reshape(B*T, H, W)
        
        # Convert masks to binary format (0 or 1)
        binary_masks = (masks > 0).float()
        
        # For BCE loss, target must have same shape as input
        if logits.dim() == 4 and binary_masks.dim() == 3:  # [B, C, H, W] vs [B, H, W]
            # Add channel dimension to binary_masks
            binary_masks_with_channel = binary_masks.unsqueeze(1)  # [B, 1, H, W]
            ce_loss = self.ce_loss(logits, binary_masks_with_channel) * self.ce_weight
        else:
            ce_loss = self.ce_loss(logits, binary_masks) * self.ce_weight
        
        # For Dice loss, convert logits to probabilities
        pred_probs = torch.sigmoid(logits)
        
        # For Dice loss, both inputs should be [B, H, W]
        if pred_probs.dim() == 4:  # [B, C, H, W]
            pred_probs_squeezed = pred_probs.squeeze(1)  # [B, H, W]
            dice_loss = self.dice_loss(pred_probs_squeezed, binary_masks) * self.dice_weight
        else:
            dice_loss = self.dice_loss(pred_probs, binary_masks) * self.dice_weight
        
        # For boundary loss, inputs should also be [B, H, W]
        if pred_probs.dim() == 4:  # [B, C, H, W]
            boundary_loss = self.boundary_loss(pred_probs_squeezed, binary_masks) * self.boundary_weight
        else:
            boundary_loss = self.boundary_loss(pred_probs, binary_masks) * self.boundary_weight
        
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
        """
        Compute boundary loss between predictions and targets.
        
        Args:
            predictions: Predicted probabilities [B, H, W]
            targets: Binary target masks [B, H, W]
            
        Returns:
            Boundary loss (1 - boundary IoU)
        """
        batch_size = predictions.size(0)
        
        # Create boundary maps
        if self.sobel_x.device != predictions.device:
            self.sobel_x = self.sobel_x.to(predictions.device)
            self.sobel_y = self.sobel_y.to(predictions.device)
        
        # Get edges from predictions - add channel dimension
        pred_boundaries = self._get_boundaries(predictions.unsqueeze(1))
        
        # Get edges from targets - add channel dimension
        target_boundaries = self._get_boundaries(targets.unsqueeze(1).float())
        
        # Calculate boundary IoU loss
        intersection = (pred_boundaries * target_boundaries).sum(dim=[1, 2, 3])
        union = pred_boundaries.sum(dim=[1, 2, 3]) + target_boundaries.sum(dim=[1, 2, 3]) - intersection
        boundary_iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return (1 - boundary_iou).mean()
    
    def _get_boundaries(self, tensor):
        """
        Extract boundaries from a mask using Sobel filters.
        
        Args:
            tensor: Input tensor [B, 1, H, W]
            
        Returns:
            Gradient magnitude of boundaries
        """
        # Apply Sobel filters for edge detection
        grad_x = F.conv2d(tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor, self.sobel_y, padding=1)
        
        # Calculate gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and return
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
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
        pred_flat = predictions.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)
        
        # Compute intersection and union
        intersection = (pred_flat * targets_flat).sum(1)
        union = pred_flat.sum(1) + targets_flat.sum(1)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice.mean()