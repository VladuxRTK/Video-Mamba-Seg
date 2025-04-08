import torch
import numpy as np
from typing import Dict, List

def compute_boundary_metrics(
    pred_masks: torch.Tensor,  # [T, C, H, W]
    gt_masks: torch.Tensor     # [T, H, W]
) -> Dict[str, float]:
    """
    Compute boundary-based F-measure metrics following DAVIS benchmark.
    
    Args:
        pred_masks: Predicted segmentation masks over time
        gt_masks: Ground truth masks over time
        
    Returns:
        Dictionary containing F-mean, F-recall, and F-decay
    """
    num_frames = pred_masks.shape[0]
    f_scores = []
    
    for t in range(num_frames):
        pred = pred_masks[t].argmax(0)
        gt = gt_masks[t]
        
        # Get boundaries
        pred_boundary = get_mask_boundary(pred)
        gt_boundary = get_mask_boundary(gt)
        
        # Compute precision and recall with tolerance
        precision = compute_boundary_precision(pred_boundary, gt_boundary)
        recall = compute_boundary_recall(pred_boundary, gt_boundary)
        
        # Compute F-measure
        f_score = 2 * precision * recall / (precision + recall + 1e-6)
        f_scores.append(f_score)
    
    f_mean = np.mean(f_scores)
    f_recall = np.mean([f > 0.5 for f in f_scores])
    f_decay = max(0, f_scores[0] - np.mean(f_scores[-4:]))
    
    return {
        'F_mean': f_mean,
        'F_recall': f_recall,
        'F_decay': f_decay
    }