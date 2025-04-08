import torch
import numpy as np
from typing import Dict, List

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth masks.
    This is also known as the Jaccard index or J-score in DAVIS benchmark.
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
    
    Returns:
        IoU score between 0 and 1
    """
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return (intersection / (union + 1e-6)).item()

def compute_region_metrics(
    pred_masks: torch.Tensor,  # [T, C, H, W]
    gt_masks: torch.Tensor     # [T, H, W]
) -> Dict[str, float]:
    """
    Compute region-based metrics following DAVIS benchmark.
    
    Args:
        pred_masks: Predicted segmentation masks over time
        gt_masks: Ground truth masks over time
    
    Returns:
        Dictionary containing J-mean, J-recall, and J-decay
    """
    num_frames = pred_masks.shape[0]
    ious = []
    
    # Compute IoU for each frame
    for t in range(num_frames):
        pred = pred_masks[t].argmax(0)  # Convert to class indices
        gt = gt_masks[t]
        ious.append(compute_iou(pred == 1, gt == 1))
    
    # Compute metrics
    j_mean = np.mean(ious)
    j_recall = np.mean([iou > 0.5 for iou in ious])  # % of frames with IoU > 0.5
    
    # Compute decay (difference between first and last frames)
    j_decay = max(0, ious[0] - np.mean(ious[-4:]))  # Following DAVIS protocol
    
    return {
        'J_mean': j_mean,
        'J_recall': j_recall,
        'J_decay': j_decay
    }
