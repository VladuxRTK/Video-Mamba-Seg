import torch
import torch.nn as nn
from typing import Dict, Optional


from .segmentation import BinarySegmentationLoss
from .temporal_consistency import TemporalConsistencyLoss

class CombinedLoss(nn.Module):
    """
    Combines binary segmentation loss with temporal consistency loss.
    This is designed for binary video segmentation rather than instance segmentation.
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        temporal_weight: float = 1.0
    ):
        super().__init__()
        self.seg_loss = BinarySegmentationLoss(ce_weight, dice_weight)
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        flows: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with proper temporal handling.
        
        Args:
            outputs: Dictionary containing 'logits' or 'pred_masks'
            targets: Dictionary containing 'masks'
            flows: Optional optical flow between frames
            
        Returns:
            Dictionary containing all loss terms and total loss
        """
        # Compute segmentation losses
        seg_losses = self.seg_loss(outputs, targets)
        
        # Initialize total losses dictionary with segmentation losses
        losses = dict(seg_losses)
        
        # Compute temporal consistency loss if needed
        if flows is not None:
            temp_losses = self.temporal_loss(
                outputs.get('pred_masks', outputs.get('logits')), 
                flows
            )
            # Add temporal losses to the total losses
            for key, value in temp_losses.items():
                losses[key] = value
        
        # Compute total loss
        losses['total_loss'] = sum(loss for name, loss in losses.items() 
                                  if name != 'total_loss')
        
        return losses