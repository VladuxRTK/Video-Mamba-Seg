import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.optimize
from scipy.ndimage import distance_transform_edt

class VideoInstanceEvaluator:
    """
    Comprehensive evaluation metrics for video instance segmentation.
    Implements J&F measure, temporal consistency, and instance stability metrics.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.scores = {
            'J': [],  # Region similarity (IoU)
            'F': [],  # Boundary similarity
            'T': [],  # Temporal stability
            'instance_stability': []  # Instance ID consistency
        }
    
    def evaluate_sequence(
        self,
        pred_masks: torch.Tensor,  # [T, N, H, W] or [T, 1, H, W]
        gt_masks: torch.Tensor,    # [T, H, W]
        flow: Optional[torch.Tensor] = None  # [T-1, 2, H, W]
    ) -> Dict[str, float]:
        """
        Evaluate a full video sequence.
        
        Args:
            pred_masks: Predicted instance masks [T, N, H, W] or binary mask [T, 1, H, W]
            gt_masks: Ground truth instance masks [T, H, W]
            flow: Optional optical flow between frames [T-1, 2, H, W]
            
        Returns:
            Dictionary with evaluation metrics
        """
        T = pred_masks.shape[0]
        assert gt_masks.shape[0] == T, "Mismatched sequence lengths"
        
        # Get unique IDs in ground truth
        gt_ids = []
        for t in range(T):
            ids = torch.unique(gt_masks[t])
            ids = ids[ids > 0]  # Skip background
            gt_ids.extend(ids.tolist())
        gt_ids = sorted(set(gt_ids))  # Unique IDs across all frames
        
        # Convert predictions to binary instance masks if needed
        if pred_masks.shape[1] == 1:  # Binary segmentation
            binary_preds = (pred_masks > 0.5).squeeze(1)  # [T, H, W]
            pred_instances = [binary_preds] * len(gt_ids)  # Treat as single instance
        else:  # Instance segmentation
            pred_instances = []
            for i in range(min(len(gt_ids), pred_masks.shape[1])):
                pred_instances.append((pred_masks[:, i] > 0.5).squeeze(-1))  # [T, H, W]
        
        # Match predicted instances to ground truth instances
        if len(pred_instances) > 0 and len(gt_ids) > 0:
            matches = self._match_instances(pred_instances, gt_masks, gt_ids)
        else:
            matches = []
        
        # Calculate metrics for each instance
        j_scores = []
        f_scores = []
        
        for pred_idx, gt_id in matches:
            # Get masks for this instance
            pred_masks_inst = pred_instances[pred_idx]  # [T, H, W]
            gt_masks_inst = (gt_masks == gt_id).float()  # [T, H, W]
            
            # Calculate J measure (IoU)
            j_score = self._compute_j_measure(pred_masks_inst, gt_masks_inst)
            j_scores.append(j_score)
            
            # Calculate F measure (boundary similarity)
            f_score = self._compute_f_measure(pred_masks_inst, gt_masks_inst)
            f_scores.append(f_score)
        
        # Calculate temporal stability
        t_score = self._compute_temporal_stability(pred_masks, flow)
        
        # Calculate instance stability
        instance_stability = self._compute_instance_stability(pred_masks)
        
        # Store scores
        self.scores['J'].extend(j_scores)
        self.scores['F'].extend(f_scores)
        self.scores['T'].append(t_score)
        self.scores['instance_stability'].append(instance_stability)
        
        # Calculate mean scores
        mean_j = np.mean(j_scores) if j_scores else 0.0
        mean_f = np.mean(f_scores) if f_scores else 0.0
        
        # Calculate J&F score
        jf_score = (mean_j + mean_f) / 2.0
        
        # Return metrics for this sequence
        return {
            'J&F': jf_score,
            'J_mean': mean_j,
            'F_mean': mean_f,
            'T_mean': t_score,
            'instance_stability': instance_stability
        }
    
    def _match_instances(
        self,
        pred_instances: List[torch.Tensor],  # List of [T, H, W]
        gt_masks: torch.Tensor,             # [T, H, W]
        gt_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Match predicted instances to ground truth instances using Hungarian algorithm.
        
        Args:
            pred_instances: List of predicted instance masks
            gt_masks: Ground truth instance masks
            gt_ids: List of ground truth instance IDs
            
        Returns:
            List of (pred_idx, gt_id) pairs representing matched instances
        """
        num_pred = len(pred_instances)
        num_gt = len(gt_ids)
        
        # If no predictions or no ground truth, return empty list
        if num_pred == 0 or num_gt == 0:
            return []
        
        # Calculate IoU between each prediction and ground truth instance
        iou_matrix = np.zeros((num_pred, num_gt))
        
        for i, pred_mask in enumerate(pred_instances):
            for j, gt_id in enumerate(gt_ids):
                gt_mask = (gt_masks == gt_id).float()
                iou = self._compute_j_measure(pred_mask, gt_mask)
                iou_matrix[i, j] = iou
        
        # Use Hungarian algorithm to find optimal matching
        # We negate IoU because the algorithm minimizes cost
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-iou_matrix)
        
        # Convert to list of (pred_idx, gt_id) pairs
        matches = [(row_ind[i], gt_ids[col_ind[i]]) for i in range(len(row_ind))]
        
        # Filter out matches with IoU below threshold
        iou_threshold = 0.1  # Low threshold to include weak matches
        matches = [(pred_idx, gt_id) for pred_idx, gt_id in matches 
                  if iou_matrix[pred_idx, gt_ids.index(gt_id)] >= iou_threshold]
        
        return matches
    
    def _compute_j_measure(
        self,
        pred_masks: torch.Tensor,  # [T, H, W]
        gt_masks: torch.Tensor     # [T, H, W]
    ) -> float:
        """
        Compute J measure (IoU) for an instance across frames.
        
        Args:
            pred_masks: Predicted masks for one instance
            gt_masks: Ground truth masks for one instance
            
        Returns:
            Mean IoU across frames
        """
        ious = []
        
        for t in range(pred_masks.shape[0]):
            pred = pred_masks[t] > 0.5
            gt = gt_masks[t] > 0.5
            
            # Skip empty frames
            if not gt.any():
                continue
            
            # Calculate IoU
            intersection = (pred & gt).sum().float()
            union = (pred | gt).sum().float()
            
            # Add small epsilon to avoid division by zero
            iou = intersection / (union + 1e-6)
            ious.append(iou.item())
        
        return np.mean(ious) if ious else 0.0
    
    def _compute_f_measure(
        self,
        pred_masks: torch.Tensor,  # [T, H, W]
        gt_masks: torch.Tensor     # [T, H, W]
    ) -> float:
        """
        Compute F measure (boundary similarity) for an instance across frames.
        
        Args:
            pred_masks: Predicted masks for one instance
            gt_masks: Ground truth masks for one instance
            
        Returns:
            Mean F measure across frames
        """
        f_scores = []
        
        for t in range(pred_masks.shape[0]):
            pred = pred_masks[t] > 0.5
            gt = gt_masks[t] > 0.5
            
            # Skip empty frames
            if not gt.any():
                continue
            
            # Calculate precision and recall based on boundary pixels
            pred_boundary = self._get_boundary(pred)
            gt_boundary = self._get_boundary(gt)
            
            # Calculate precision and recall
            precision = (pred_boundary & gt_boundary).sum().float() / (pred_boundary.sum().float() + 1e-6)
            recall = (pred_boundary & gt_boundary).sum().float() / (gt_boundary.sum().float() + 1e-6)
            
            # Calculate F measure
            f_score = (2 * precision * recall) / (precision + recall + 1e-6)
            f_scores.append(f_score.item())
        
        return np.mean(f_scores) if f_scores else 0.0
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Get boundary of a binary mask.
        
        Args:
            mask: Binary mask tensor
            
        Returns:
            Binary boundary mask
        """
        # Convert to numpy for morphological operations
        mask_np = mask.cpu().numpy()
        
        # Calculate distance transform
        dist = distance_transform_edt(mask_np)
        
        # Get boundary pixels (distance = 1)
        boundary = (dist <= 1) & (dist > 0)
        
        # Convert back to tensor
        return torch.from_numpy(boundary).to(mask.device)
    
    def _compute_temporal_stability(
        self,
        pred_masks: torch.Tensor,  # [T, N, H, W] or [T, 1, H, W]
        flow: Optional[torch.Tensor] = None  # [T-1, 2, H, W]
    ) -> float:
        """
        Compute temporal stability score.
        
        Args:
            pred_masks: Predicted instance masks
            flow: Optional optical flow between frames
            
        Returns:
            Temporal stability score
        """
        T = pred_masks.shape[0]
        
        # If only one frame, perfect stability
        if T <= 1:
            return 1.0
        
        # Calculate frame-to-frame changes
        stability_scores = []
        
        for t in range(T - 1):
            curr_mask = pred_masks[t]
            next_mask = pred_masks[t+1]
            
            # If using flow for motion-compensated evaluation
            if flow is not None:
                # Warp current mask to next frame using flow
                curr_flow = flow[t]  # [2, H, W]
                warped_mask = self._warp_mask(curr_mask, curr_flow)
                
                # Calculate stability between warped mask and next mask
                stability = 1.0 - torch.abs(warped_mask - next_mask).mean()
            else:
                # Simple temporal consistency without flow
                stability = 1.0 - torch.abs(curr_mask - next_mask).mean()
            
            stability_scores.append(stability.item())
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _warp_mask(
        self,
        mask: torch.Tensor,  # [N, H, W] or [1, H, W]
        flow: torch.Tensor   # [2, H, W]
    ) -> torch.Tensor:
        """
        Warp mask using optical flow.
        
        Args:
            mask: Mask to warp
            flow: Optical flow field
            
        Returns:
            Warped mask
        """
        # Get mask dimensions
        if mask.dim() == 3:
            N, H, W = mask.shape
        else:
            H, W = mask.shape
            N = 1
            mask = mask.unsqueeze(0)
        
        # Create sampling grid from flow
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=mask.device),
            torch.arange(W, device=mask.device),
            indexing='ij'
        )
        
        # Add flow to grid
        grid_x = grid_x + flow[0]
        grid_y = grid_y + flow[1]
        
        # Normalize grid coordinates to [-1, 1]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        # Stack coordinates
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Reshape mask for grid_sample
        mask_flat = mask.view(1, N, H, W)
        
        # Warp mask using grid sample
        warped_mask = torch.nn.functional.grid_sample(
            mask_flat,
            grid.unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Return warped mask in original shape
        return warped_mask.view(N, H, W)
    
    def _compute_instance_stability(self, pred_masks: torch.Tensor) -> float:
        """
        Compute instance stability score by measuring how consistently
        the model maintains instance identity across frames.
        
        Args:
            pred_masks: Predicted instance masks [T, N, H, W]
            
        Returns:
            Instance stability score
        """
        T, N, H, W = pred_masks.shape
        
        # If only one frame or one instance, perfect stability
        if T <= 1 or N <= 1:
            return 1.0
        
        # Calculate instance consistency across frames
        consistency_matrix = torch.zeros((N, N), device=pred_masks.device)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Calculate temporal IoU between instances i and j
                iou_sequence = []
                
                for t in range(T-1):
                    mask_i_t = pred_masks[t, i] > 0.5
                    mask_j_t1 = pred_masks[t+1, j] > 0.5
                    
                    # Calculate IoU
                    intersection = (mask_i_t & mask_j_t1).sum().float()
                    union = (mask_i_t | mask_j_t1).sum().float()
                    
                    # Add small epsilon to avoid division by zero
                    iou = intersection / (union + 1e-6)
                    iou_sequence.append(iou.item())
                
                # Average IoU between instances i and j across frames
                consistency_matrix[i, j] = np.mean(iou_sequence) if iou_sequence else 0.0
        
        # Calculate stability score
        # Lower IoU between different instances means better separation
        # We take 1 - average IoU between different instances
        stability = 1.0 - consistency_matrix.mean().item()
        
        return stability
    
    def get_global_metrics(self) -> Dict[str, float]:
        """
        Get global metrics averaged across all evaluated sequences.
        
        Returns:
            Dictionary with global metrics
        """
        # Calculate mean scores
        j_mean = np.mean(self.scores['J']) if self.scores['J'] else 0.0
        f_mean = np.mean(self.scores['F']) if self.scores['F'] else 0.0
        t_mean = np.mean(self.scores['T']) if self.scores['T'] else 1.0
        instance_stability = np.mean(self.scores['instance_stability']) if self.scores['instance_stability'] else 1.0
        
        # Calculate J&F score
        jf_score = (j_mean + f_mean) / 2.0
        
        return {
            'J&F': jf_score,
            'J_mean': j_mean,
            'F_mean': f_mean,
            'T_mean': t_mean,
            'instance_stability': instance_stability
        }


class DAVISEvaluator:
    """
    DAVIS benchmark evaluator for video object segmentation.
    Implements the official DAVIS evaluation protocol.
    """
    def __init__(self):
        self.instance_evaluator = VideoInstanceEvaluator()
    
    def evaluate(
        self,
        predictions: List[torch.Tensor],  # List of [T, N, H, W]
        ground_truths: List[torch.Tensor],  # List of [T, H, W]
        sequence_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple sequences using DAVIS protocol.
        
        Args:
            predictions: List of predicted masks for each sequence
            ground_truths: List of ground truth masks for each sequence
            sequence_names: Names of the sequences
            
        Returns:
            Dictionary with per-sequence and global metrics
        """
        # Reset evaluator
        self.instance_evaluator.reset()
        
        # Evaluate each sequence
        sequence_metrics = {}
        
        for pred, gt, name in zip(predictions, ground_truths, sequence_names):
            metrics = self.instance_evaluator.evaluate_sequence(pred, gt)
            sequence_metrics[name] = metrics
        
        # Get global metrics
        global_metrics = self.instance_evaluator.get_global_metrics()
        
        # Return all metrics
        return {
            'global': global_metrics,
            'sequences': sequence_metrics
        }
    # In utils/evaluation.py, update or add this method to DAVISEvaluator

    def evaluate_binary_segmentation(
        self,
        predictions: List[torch.Tensor],  # List of [T, 1, H, W]
        ground_truths: List[torch.Tensor],  # List of [T, H, W]
        sequence_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate binary segmentation performance on DAVIS.
        
        Args:
            predictions: List of predicted binary masks
            ground_truths: List of ground truth masks
            sequence_names: Names of the sequences
            
        Returns:
            Dictionary with evaluation metrics
        """
        sequence_metrics = {}
        global_j_scores = []
        global_f_scores = []
        global_t_scores = []
        
        for pred, gt, name in zip(predictions, ground_truths, sequence_names):
            # Convert predictions to binary
            if pred.dim() == 4:  # [T, 1, H, W]
                pred = pred.squeeze(1)
            binary_pred = (pred > 0.5).bool()
            
            # Convert ground truth to binary
            binary_gt = (gt > 0).bool()
            
            # Compute J measure (IoU)
            j_scores = []
            for t in range(pred.shape[0]):
                intersection = (binary_pred[t] & binary_gt[t]).float().sum()
                union = (binary_pred[t] | binary_gt[t]).float().sum()
                iou = (intersection / (union + 1e-6)).item()
                j_scores.append(iou)
            
            j_mean = np.mean(j_scores)
            global_j_scores.extend(j_scores)
            
            # Compute F measure (boundary precision)
            f_scores = []
            for t in range(pred.shape[0]):
                pred_boundary = self._get_boundary(binary_pred[t])
                gt_boundary = self._get_boundary(binary_gt[t])
                
                precision = (pred_boundary & gt_boundary).float().sum() / (pred_boundary.float().sum() + 1e-6)
                recall = (pred_boundary & gt_boundary).float().sum() / (gt_boundary.float().sum() + 1e-6)
                
                f_score = (2 * precision * recall / (precision + recall + 1e-6)).item()
                f_scores.append(f_score)
            
            f_mean = np.mean(f_scores)
            global_f_scores.extend(f_scores)
            
            # Compute temporal stability
            t_scores = []
            for t in range(pred.shape[0] - 1):
                stability = 1.0 - (binary_pred[t] ^ binary_pred[t+1]).float().mean().item()
                t_scores.append(stability)
            
            t_mean = np.mean(t_scores) if t_scores else 1.0
            global_t_scores.extend(t_scores)
            
            # Store sequence metrics
            sequence_metrics[name] = {
                'J_mean': j_mean,
                'F_mean': f_mean,
                'T_mean': t_mean,
                'J&F': (j_mean + f_mean) / 2
            }
        
        # Compute global metrics
        global_metrics = {
            'J_mean': np.mean(global_j_scores),
            'F_mean': np.mean(global_f_scores),
            'T_mean': np.mean(global_t_scores),
            'J&F': (np.mean(global_j_scores) + np.mean(global_f_scores)) / 2
        }
        
        return {
            'global': global_metrics,
            'sequences': sequence_metrics
        }

    def _get_boundary(self, mask):
        """Helper method to extract boundary pixels from a mask."""
        # Implementation depends on your preference
        dilated = torch.nn.functional.max_pool2d(
            mask.float().unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0) > 0.5
        eroded = torch.nn.functional.avg_pool2d(
            mask.float().unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0) >= 0.9
        
        return dilated & (~eroded)
    
    def print_results(self, results: Dict[str, Dict[str, float]]):
        """
        Print evaluation results in a readable format.
        
        Args:
            results: Dictionary with evaluation results
        """
        global_metrics = results['global']
        sequence_metrics = results['sequences']
        
        print("\n" + "="*50)
        print(f"DAVIS Evaluation Results")
        print("="*50)
        
        # Print global metrics
        print("\nGlobal Metrics:")
        print(f"J&F: {global_metrics['J&F']:.4f}")
        print(f"J-Mean: {global_metrics['J_mean']:.4f}")
        print(f"F-Mean: {global_metrics['F_mean']:.4f}")
        print(f"Temporal Stability: {global_metrics['T_mean']:.4f}")
        print(f"Instance Stability: {global_metrics['instance_stability']:.4f}")
        
        # Print per-sequence metrics
        print("\nPer-Sequence Metrics:")
        for name, metrics in sequence_metrics.items():
            print(f"\n{name}:")
            print(f"  J&F: {metrics['J&F']:.4f}")
            print(f"  J-Mean: {metrics['J_mean']:.4f}")
            print(f"  F-Mean: {metrics['F_mean']:.4f}")
            print(f"  Temporal Stability: {metrics['T_mean']:.4f}")
            print(f"  Instance Stability: {metrics['instance_stability']:.4f}")
        
        print("\n" + "="*50)