class DAVISEvaluator:
    """
    Main evaluator class that combines all DAVIS benchmark metrics.
    """
    def __init__(self):
        pass
    
    def evaluate_sequence(
        self,
        pred_masks: torch.Tensor,  # [T, C, H, W]
        gt_masks: torch.Tensor     # [T, H, W]
    ) -> Dict[str, float]:
        """
        Evaluate a single video sequence using all metrics.
        """
        metrics = {}
        
        # Region similarity (J)
        metrics.update(compute_region_metrics(pred_masks, gt_masks))
        
        # Boundary accuracy (F)
        metrics.update(compute_boundary_metrics(pred_masks, gt_masks))
        
        # Temporal stability (T)
        metrics.update(compute_temporal_metrics(pred_masks))
        
        return metrics
    
    def evaluate_dataset(
        self,
        predictions: List[torch.Tensor],
        ground_truths: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset and compute global metrics.
        """
        all_metrics = []
        
        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate_sequence(pred, gt)
            all_metrics.append(metrics)
        
        # Average across sequences
        global_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            global_metrics[key] = np.mean(values)
        
        return global_metrics