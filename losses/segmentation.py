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
        self.boundary_loss = BoundaryLoss() if boundary_weight > 0 else None
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for binary segmentation.
        
        Args:
            outputs: Dictionary containing 'logits' or 'pred_masks'
            targets: Dictionary containing 'masks'
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Get logits and reshape if necessary
        logits = outputs['logits']  # [B, T, 1, H, W]
        if logits.dim() == 5:
            B, T = logits.shape[:2]
            logits = logits.view(B*T, 1, logits.shape[3], logits.shape[4])
        
        # Get target masks and reshape if necessary
        masks = targets['masks']  # [B, T, H, W]
        if masks.dim() == 4:
            B, T = masks.shape[:2]
            masks = masks.view(B*T, masks.shape[2], masks.shape[3])
        
        # Convert masks to binary format (0 or 1)
        binary_masks = (masks > 0).float()
        
        # Add channel dimension to binary_masks
        binary_masks_with_channel = binary_masks.unsqueeze(1)  # [B*T, 1, H, W]
        
        # Check if dimensions match - if not, resize logits to match target
        if logits.shape[2:] != binary_masks_with_channel.shape[2:]:
            logits = F.interpolate(
                logits, 
                size=binary_masks_with_channel.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # Check for NaN values in logits and replace them
        if torch.isnan(logits).any():
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            print("WARNING: NaN values found in logits, replacing with zeros")
        
        # Calculate BCE loss
        ce_loss = self.ce_loss(logits, binary_masks_with_channel) * self.ce_weight
        
        # Calculate Dice loss (also resize pred_probs if needed)
        pred_probs = torch.sigmoid(logits)
        
        # Check for NaN values in pred_probs
        if torch.isnan(pred_probs).any():
            pred_probs = torch.where(torch.isnan(pred_probs), torch.zeros_like(pred_probs), pred_probs)
            print("WARNING: NaN values found in pred_probs, replacing with zeros")
        
        dice_loss = self.dice_loss(pred_probs.squeeze(1), binary_masks) * self.dice_weight
        
        # Calculate Boundary loss if enabled
        boundary_loss = 0.0
        if self.boundary_loss is not None:
            boundary_loss = self.boundary_loss(pred_probs.squeeze(1), binary_masks) * self.boundary_weight
        
        # Calculate total loss
        total_loss = ce_loss + dice_loss + boundary_loss
        
        # Check for NaN in final loss
        if torch.isnan(total_loss):
            print(f"WARNING: NaN in loss - CE: {ce_loss.item()}, Dice: {dice_loss.item()}, Boundary: {boundary_loss}")
            # Provide a fallback loss to prevent training breakdown
            total_loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss if self.boundary_loss is not None else 0.0
        }

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