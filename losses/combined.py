import torch
import torch.nn as nn
from typing import Dict, Optional
import os
from datetime import datetime

from .segmentation import BinarySegmentationLoss
from .temporal_consistency import TemporalConsistencyLoss

class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced combined loss with temporal consistency for stable training.
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 2.0,
        temporal_weight: float = 0.5,
        boundary_weight: float = 0.0
    ):
        super().__init__()
        self.seg_loss = BinarySegmentationLoss(ce_weight, dice_weight, boundary_weight)
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
        
        # Enhanced logging
        self.debug_log_path = "logs/combined_loss_debug.txt"
        os.makedirs("logs", exist_ok=True)
        
        with open(self.debug_log_path, "w") as f:
            f.write(f"=== COMBINED LOSS DEBUG LOG STARTED: {datetime.now()} ===\n\n")
        
        self.batch_count = 0
    
    def _log_debug(self, message: str):
        """Log debug message to file."""
        with open(self.debug_log_path, "a") as f:
            f.write(f"{message}\n")
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        flows: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with temporal consistency.
        
        Args:
            outputs: Dictionary containing 'logits' and 'pred_masks'
            targets: Dictionary containing 'masks'
            flows: Optional optical flow between frames
            
        Returns:
            Dictionary containing all loss terms and total loss
        """
        self.batch_count += 1
        
        # Compute segmentation losses
        seg_losses = self.seg_loss(outputs, targets)
        
        # Initialize total losses dictionary
        losses = dict(seg_losses)
        
        # Compute temporal consistency loss
        if 'pred_masks' in outputs:
            temp_losses = self.temporal_loss(outputs['pred_masks'], flows)
            
            # Add temporal losses
            for key, value in temp_losses.items():
                losses[key] = value
        
        # Compute total loss
        total_loss = sum(loss for name, loss in losses.items() 
                        if name != 'total_loss')
        losses['total_loss'] = total_loss
        
        # Enhanced logging every 100 batches
        if self.batch_count % 100 == 0:
            self._log_debug(f"\nBATCH {self.batch_count} - COMBINED LOSS SUMMARY:")
            self._log_debug(f"  Total Loss: {total_loss.item():.6f}")
            for key, value in losses.items():
                if key != 'total_loss':
                    loss_val = value.item() if hasattr(value, 'item') else value
                    self._log_debug(f"  {key}: {loss_val:.6f}")
        
        return losses