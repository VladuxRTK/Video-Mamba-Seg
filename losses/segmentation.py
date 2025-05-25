import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging
import os
from datetime import datetime

class BinarySegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 1.5, boundary_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = ImprovedDiceLoss()
        self.boundary_loss = BoundaryLoss() if boundary_weight > 0 else None
        
        # 🚀 CREATE DEBUG LOG FILE
        self.debug_log_path = "logs/loss_debug.txt"
        os.makedirs("logs", exist_ok=True)
        
        # Initialize debug log
        with open(self.debug_log_path, "w") as f:
            f.write(f"=== LOSS DEBUG LOG STARTED: {datetime.now()} ===\n\n")
        
        self.batch_count = 0
    
    def _log_debug(self, message: str):
        """Log debug message to file and optionally print."""
        with open(self.debug_log_path, "a") as f:
            f.write(f"{message}\n")
        
        # Also print to console (but this gets saved in the log file)
        print(message)
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for binary segmentation with detailed logging.
        """
        self.batch_count += 1
        
        # Get logits and masks
        logits = outputs['logits']  # [B, T, 1, H, W]
        masks = targets['masks']  # [B, T, H, W]
        
        # 🚀 LOG TO FILE INSTEAD OF JUST PRINTING
        self._log_debug(f"\n{'='*60}")
        self._log_debug(f"BATCH {self.batch_count} - LOSS DEBUG")
        self._log_debug(f"{'='*60}")
        self._log_debug(f"Input logits shape: {logits.shape}")
        self._log_debug(f"Input masks shape: {masks.shape}")
        
        # Flatten temporal and batch dimensions
        if logits.dim() == 5:  # [B, T, 1, H, W]
            B, T, C, H, W = logits.shape
            logits_flat = logits.reshape(B * T, C, H, W)
        else:
            logits_flat = logits
            
        if masks.dim() == 4:  # [B, T, H, W]
            B, T, H, W = masks.shape
            masks_flat = masks.reshape(B * T, H, W)
        else:
            masks_flat = masks
            
        self._log_debug(f"Flattened logits shape: {logits_flat.shape}")
        self._log_debug(f"Flattened masks shape: {masks_flat.shape}")
        
        # Convert masks to binary format
        binary_masks = (masks_flat > 0).float()
        
        # Resize logits if needed
        if logits_flat.shape[2:] != binary_masks.shape[1:]:
            self._log_debug(f"Resizing logits from {logits_flat.shape[2:]} to {binary_masks.shape[1:]}")
            logits_flat = F.interpolate(
                logits_flat, 
                size=binary_masks.shape[1:],
                mode='bilinear', 
                align_corners=False
            )
        
        # Prepare targets for BCE
        binary_masks_with_channel = binary_masks.unsqueeze(1)
        
        self._log_debug(f"Final logits shape: {logits_flat.shape}")
        self._log_debug(f"Final BCE targets shape: {binary_masks_with_channel.shape}")
        self._log_debug(f"Final Dice targets shape: {binary_masks.shape}")
        
        # Calculate BCE loss
        ce_loss = self.ce_loss(logits_flat, binary_masks_with_channel) * self.ce_weight
        
        # Calculate probabilities for Dice loss
        pred_probs = torch.sigmoid(logits_flat)
        pred_probs_flat = pred_probs.squeeze(1)
        
        self._log_debug(f"Pred probs shape: {pred_probs_flat.shape}")
        self._log_debug(f"Pred probs range: [{pred_probs_flat.min():.4f}, {pred_probs_flat.max():.4f}]")
        self._log_debug(f"Binary masks range: [{binary_masks.min():.4f}, {binary_masks.max():.4f}]")
        
        # Calculate Dice loss
        dice_loss = self.dice_loss(pred_probs_flat, binary_masks) * self.dice_weight
        
        # Calculate Boundary loss if enabled
        boundary_loss = 0.0
        if self.boundary_loss is not None:
            boundary_loss = self.boundary_loss(pred_probs_flat, binary_masks) * self.boundary_weight
        
        # Calculate total loss
        total_loss = ce_loss + dice_loss + boundary_loss
        
        # 🚀 LOG FINAL RESULTS
        self._log_debug(f"🎯 LOSS RESULTS:")
        self._log_debug(f"   CE Loss: {ce_loss.item():.6f}")
        self._log_debug(f"   Dice Loss: {dice_loss.item():.6f}")
        self._log_debug(f"   Boundary Loss: {boundary_loss:.6f}")
        self._log_debug(f"   Total Loss: {total_loss.item():.6f}")
        
        # Handle NaN gracefully
        if torch.isnan(total_loss):
            self._log_debug("❌ WARNING: NaN in total loss!")
            if not torch.isnan(ce_loss):
                total_loss = ce_loss
                self._log_debug("   Using CE loss only")
            else:
                total_loss = torch.tensor(0.1, device=logits.device, requires_grad=True)
                self._log_debug("   Using fallback loss")
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss if self.boundary_loss is not None else 0.0
        }


class ImprovedDiceLoss(nn.Module):
    """Improved Dice loss with file logging."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.debug_log_path = "logs/dice_debug.txt"
        os.makedirs("logs", exist_ok=True)
        
        # Initialize dice debug log
        with open(self.debug_log_path, "w") as f:
            f.write(f"=== DICE DEBUG LOG STARTED: {datetime.now()} ===\n\n")
        
        self.call_count = 0
    
    def _log_debug(self, message: str):
        """Log debug message to dice-specific file."""
        with open(self.debug_log_path, "a") as f:
            f.write(f"{message}\n")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [N, H, W] - probabilities between 0 and 1
            targets: [N, H, W] - binary targets (0 or 1)
        """
        self.call_count += 1
        
        self._log_debug(f"\nDICE CALL {self.call_count}:")
        self._log_debug(f"  Pred shape: {predictions.shape}, Target shape: {targets.shape}")
        
        # Ensure shapes match exactly
        assert predictions.shape == targets.shape, f"Shape mismatch: {predictions.shape} vs {targets.shape}"
        
        # Flatten
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # Calculate intersection and sums
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        self._log_debug(f"  Intersection: {intersection.item():.4f}")
        self._log_debug(f"  Pred sum: {pred_sum.item():.4f}")
        self._log_debug(f"  Target sum: {target_sum.item():.4f}")
        
        # Calculate Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1. - dice_coeff
        
        self._log_debug(f"  Dice coefficient: {dice_coeff.item():.6f}")
        self._log_debug(f"  Dice loss: {dice_loss.item():.6f}")
        
        return dice_loss


# Keep your existing DiceLoss and BoundaryLoss classes as well
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = predictions.size(0)
        pred_flat = predictions.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)
        intersection = (pred_flat * targets_flat).sum(1)
        union = pred_flat.sum(1) + targets_flat.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        if self.sobel_x.device != predictions.device:
            self.sobel_x = self.sobel_x.to(predictions.device)
            self.sobel_y = self.sobel_y.to(predictions.device)
        
        pred_boundaries = self._get_boundaries(predictions.unsqueeze(1))
        target_boundaries = self._get_boundaries(targets.unsqueeze(1).float())
        
        intersection = (pred_boundaries * target_boundaries).sum(dim=[1, 2, 3])
        union = pred_boundaries.sum(dim=[1, 2, 3]) + target_boundaries.sum(dim=[1, 2, 3]) - intersection
        boundary_iou = (intersection + self.smooth) / (union + self.smooth)
        
        return (1 - boundary_iou).mean()
    
    def _get_boundaries(self, tensor):
        grad_x = F.conv2d(tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        return grad_mag
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

