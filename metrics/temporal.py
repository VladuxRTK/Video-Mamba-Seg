import torch
import numpy as np
from typing import Dict, List


def compute_temporal_metrics(
    pred_masks: torch.Tensor,  # [T, C, H, W]
) -> Dict[str, float]:
    """
    Compute temporal stability metrics following DAVIS benchmark.
    
    Args:
        pred_masks: Predicted segmentation masks over time
        
    Returns:
        Dictionary containing temporal stability metrics
    """
    num_frames = pred_masks.shape[0]
    
    # Convert to binary masks
    pred_masks = pred_masks.argmax(1)  # [T, H, W]
    
    # Compute frame-to-frame changes
    changes = []
    for t in range(num_frames - 1):
        change = (pred_masks[t+1] != pred_masks[t]).float().mean()
        changes.append(change.item())
    
    # Lower values indicate better stability
    t_mean = 1.0 - np.mean(changes)
    
    return {'T_mean': t_mean}
