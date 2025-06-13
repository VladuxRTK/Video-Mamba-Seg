# Enhanced evaluation integration for your existing codebase
# Add this to your utils/evaluation.py or create a new enhanced_evaluation.py

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

class EnhancedDAVISEvaluator:
    """
    Enhanced DAVIS evaluator with comprehensive temporal stability analysis.
    Integrates with your existing VideoMamba evaluation pipeline.
    """
    
    def __init__(self, save_visualizations: bool = True, vis_dir: str = "evaluation_results"):
        from utils.evaluation import DAVISEvaluator
        self.davis_evaluator = DAVISEvaluator()
        
        # Initialize temporal stability analyzer
        from temporal_stability_davis import TemporalStabilityAnalyzer
        self.temporal_analyzer = TemporalStabilityAnalyzer(save_dir=vis_dir)
        
        self.save_visualizations = save_visualizations
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_comprehensive(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation including DAVIS metrics + temporal stability.
        """
        model.eval()
        
        # Storage for all predictions and metrics
        all_predictions = []
        all_ground_truths = []
        all_sequences = []
        temporal_metrics_per_sequence = {}
        
        print("Running comprehensive evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                frames = batch['frames'].to(device)
                gt_masks = batch['masks'].to(device)
                sequence_name = batch.get('sequence', [f"seq_{batch_idx}"])[0]
                
                # Get model predictions
                outputs = model(frames)
                pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                
                # Store for DAVIS evaluation
                all_predictions.append(pred_masks[0].cpu())
                all_ground_truths.append(gt_masks[0].cpu())
                all_sequences.append(sequence_name)
                
                # Compute temporal stability for this sequence
                temporal_metrics = self.temporal_analyzer.compute_temporal_stability(
                    pred_masks[0].cpu().squeeze(),
                    gt_masks[0].cpu()
                )
                temporal_metrics_per_sequence[sequence_name] = temporal_metrics
                
                # Create visualizations if enabled
                if self.save_visualizations:
                    # Temporal stability visualization
                    self.temporal_analyzer.visualize_temporal_stability(
                        pred_masks[0].cpu().squeeze(),
                        sequence_name,
                        gt_masks[0].cpu()
                    )
                    
                    print(f"Sequence {sequence_name}:")
                    print(f"  T-score: {temporal_metrics['T_mean']:.4f}")
                    print(f"  T-recall: {temporal_metrics['T_recall']:.4f}")
                    print(f"  T-decay: {temporal_metrics['T_decay']:.4f}")
        
        # Run standard DAVIS evaluation
        davis_results = self.davis_evaluator.evaluate_binary_segmentation(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            sequence_names=all_sequences
        )
        
        # Compute global temporal metrics
        global_temporal_metrics = self._compute_global_temporal_metrics(
            temporal_metrics_per_sequence
        )
        
        # Combine results
        comprehensive_results = {
            'davis_metrics': davis_results['global'],
            'temporal_metrics': global_temporal_metrics,
            'per_sequence_davis': davis_results['sequences'],
            'per_sequence_temporal': temporal_metrics_per_sequence
        }
        
        # Print summary
        self._print_comprehensive_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _compute_global_temporal_metrics(
        self, 
        per_sequence_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute global temporal metrics across all sequences."""
        if not per_sequence_metrics:
            return {}
        
        # Aggregate metrics across sequences
        all_t_means = [metrics['T_mean'] for metrics in per_sequence_metrics.values()]
        all_t_recalls = [metrics['T_recall'] for metrics in per_sequence_metrics.values()]
        all_t_decays = [metrics['T_decay'] for metrics in per_sequence_metrics.values()]
        
        return {
            'global_T_mean': np.mean(all_t_means),
            'global_T_recall': np.mean(all_t_recalls),
            'global_T_decay': np.mean(all_t_decays),
            'T_std': np.std(all_t_means),
            'sequences_with_high_stability': sum(1 for t in all_t_means if t > 0.95),
            'sequences_with_low_stability': sum(1 for t in all_t_means if t < 0.85)
        }
    
    def _print_comprehensive_summary(self, results: Dict):
        """Print comprehensive evaluation summary."""
        print("\n" + "="*60)
        print("COMPREHENSIVE VIDEO SEGMENTATION EVALUATION")
        print("="*60)
        
        # DAVIS metrics
        davis = results['davis_metrics']
        print("\nDAVIS Benchmark Metrics:")
        print(f"  J&F Score: {davis.get('J&F', 0):.4f}")
        print(f"  J-Mean (IoU): {davis.get('J_mean', 0):.4f}")
        print(f"  F-Mean (Boundary): {davis.get('F_mean', 0):.4f}")
        
        # Temporal metrics
        temporal = results['temporal_metrics']
        print("\nTemporal Stability Metrics:")
        print(f"  Global T-Mean: {temporal.get('global_T_mean', 0):.4f}")
        print(f"  Global T-Recall: {temporal.get('global_T_recall', 0):.4f}")
        print(f"  Global T-Decay: {temporal.get('global_T_decay', 0):.4f}")
        print(f"  T-Score Std Dev: {temporal.get('T_std', 0):.4f}")
        
        # Stability analysis
        high_stability = temporal.get('sequences_with_high_stability', 0)
        low_stability = temporal.get('sequences_with_low_stability', 0)
        total_sequences = len(results['per_sequence_temporal'])
        
        print(f"\nStability Analysis:")
        print(f"  High Stability (T>0.95): {high_stability}/{total_sequences}")
        print(f"  Low Stability (T<0.85): {low_stability}/{total_sequences}")
        
        # Overall assessment
        print("\nOverall Assessment:")
        overall_score = (davis.get('J&F', 0) + temporal.get('global_T_mean', 0)) / 2
        print(f"  Combined Score (J&F + T): {overall_score:.4f}")
        
        if temporal.get('global_T_mean', 0) > 0.95:
            stability_assessment = "Excellent temporal stability"
        elif temporal.get('global_T_mean', 0) > 0.90:
            stability_assessment = "Good temporal stability"
        else:
            stability_assessment = "Poor temporal stability - needs improvement"
        
        print(f"  Stability Assessment: {stability_assessment}")
        
        print("="*60)


# Modified evaluation script that uses the enhanced evaluator
def evaluate_videomamba_comprehensive(
    model_path: str,
    config_path: str,
    dataset_split: str = 'val',
    save_visualizations: bool = True,
    specific_sequence: str = None
):
    """
    Comprehensive evaluation of VideoMamba including temporal stability.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        dataset_split: Dataset split to evaluate ('val', 'test')
        save_visualizations: Whether to save temporal stability visualizations
        specific_sequence: Optional specific sequence to evaluate
    """
    import yaml
    import torch
    from models.binary_mamba_segmentation import build_model
    from datasets.davis import build_davis_dataloader
    from datasets.transforms import VideoSequenceAugmentation
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {model_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataloader
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        train=False
    )
    
    dataloader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split=dataset_split,
        batch_size=1,
        transform=transform,
        specific_sequence=specific_sequence,
        **{k: v for k, v in config['dataset'].items() if k not in ['batch_size']}
    )
    
    # Create enhanced evaluator
    vis_dir = f"evaluation_results_{dataset_split}"
    if specific_sequence:
        vis_dir += f"_{specific_sequence}"
    
    evaluator = EnhancedDAVISEvaluator(
        save_visualizations=save_visualizations,
        vis_dir=vis_dir
    )
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_comprehensive(model, dataloader, device)
    
    # Save results to file
    import json
    results_file = Path(vis_dir) / "comprehensive_results.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # Save comprehensive results
    json_results = convert_for_json(results)
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualizations saved to: {vis_dir}")
    
    return results


# Integration with your existing training pipeline
def add_temporal_stability_to_trainer():
    """
    Example of how to integrate temporal stability into your existing Trainer class.
    Add this method to your utils/training.py Trainer class.
    """
    
    def validate_with_temporal_stability(self, val_loader):
        """Enhanced validation with temporal stability analysis."""
        # Run standard validation
        standard_metrics = self.validate(val_loader)
        
        # Add temporal stability analysis
        if hasattr(self, 'temporal_analyzer'):
            temporal_metrics = {}
            
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 5:  # Limit to first 5 sequences for efficiency
                        break
                        
                    frames = batch['frames'].to(self.device)
                    gt_masks = batch['masks'].to(self.device)
                    sequence_name = batch.get('sequence', [f"seq_{batch_idx}"])[0]
                    
                    outputs = self.model(frames)
                    pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                    
                    # Compute temporal stability
                    from temporal_stability_davis import TemporalStabilityAnalyzer
                    analyzer = TemporalStabilityAnalyzer()
                    
                    temp_metrics = analyzer.compute_temporal_stability(
                        pred_masks[0].cpu().squeeze(),
                        gt_masks[0].cpu()
                    )
                    
                    temporal_metrics[sequence_name] = temp_metrics['T_mean']
            
            # Add average temporal stability to metrics
            if temporal_metrics:
                avg_temporal_stability = np.mean(list(temporal_metrics.values()))
                standard_metrics['temporal_stability'] = avg_temporal_stability
                
                self.logger.info(f"Average Temporal Stability: {avg_temporal_stability:.4f}")
        
        return standard_metrics


# Command-line script for comprehensive evaluation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive VideoMamba Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], 
                       help='Dataset split to evaluate')
    parser.add_argument('--sequence', type=str, default=None, 
                       help='Specific sequence to evaluate (optional)')
    parser.add_argument('--no-vis', action='store_true', 
                       help='Disable visualization saving')
    
    args = parser.parse_args()
    
    # Run comprehensive evaluation
    results = evaluate_videomamba_comprehensive(
        model_path=args.model,
        config_path=args.config,
        dataset_split=args.split,
        save_visualizations=not args.no_vis,
        specific_sequence=args.sequence
    )
    
    # Print final summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Overall J&F Score: {results['davis_metrics'].get('J&F', 0):.4f}")
    print(f"Overall T-Score: {results['temporal_metrics'].get('global_T_mean', 0):.4f}")
    print(f"Combined Performance: {(results['davis_metrics'].get('J&F', 0) + results['temporal_metrics'].get('global_T_mean', 0))/2:.4f}")


# Utility function to quickly analyze a single sequence
def quick_temporal_analysis(
    model_path: str,
    config_path: str,
    sequence_name: str,
    save_dir: str = "quick_analysis"
):
    """
    Quick temporal stability analysis for a single sequence.
    Useful for debugging and detailed sequence analysis.
    """
    from temporal_stability_davis import evaluate_temporal_stability_on_davis
    
    print(f"Analyzing temporal stability for sequence: {sequence_name}")
    
    results = evaluate_temporal_stability_on_davis(
        model_path=model_path,
        config_path=config_path,
        sequence_name=sequence_name,
        save_dir=save_dir
    )
    
    if results:
        print(f"\nTemporal Stability Results for {sequence_name}:")
        print(f"  T-mean: {results['T_mean']:.4f}")
        print(f"  T-recall: {results['T_recall']:.4f}")
        print(f"  T-decay: {results['T_decay']:.4f}")
        
        # Interpretation
        if results['T_mean'] > 0.95:
            print("  → Excellent temporal stability!")
        elif results['T_mean'] > 0.90:
            print("  → Good temporal stability")
        elif results['T_mean'] > 0.80:
            print("  → Moderate temporal stability")
        else:
            print("  → Poor temporal stability - needs improvement")
    
    return results


# Helper function to compare temporal stability across different models
def compare_temporal_stability(
    model_paths: List[str],
    model_names: List[str],
    config_path: str,
    sequence_name: str = None
):
    """
    Compare temporal stability across different model checkpoints.
    """
    results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating {model_name}...")
        
        try:
            if sequence_name:
                seq_results = quick_temporal_analysis(
                    model_path, config_path, sequence_name, 
                    save_dir=f"comparison_{model_name}"
                )
            else:
                seq_results = evaluate_videomamba_comprehensive(
                    model_path, config_path, save_visualizations=False
                )
                seq_results = seq_results['temporal_metrics']
            
            results[model_name] = seq_results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = None
    
    # Print comparison
    print("\n" + "="*60)
    print("TEMPORAL STABILITY COMPARISON")
    print("="*60)
    
    for model_name, result in results.items():
        if result:
            if 'T_mean' in result:
                t_score = result['T_mean']
            else:
                t_score = result.get('global_T_mean', 0)
            
            print(f"{model_name:20s}: T-score = {t_score:.4f}")
        else:
            print(f"{model_name:20s}: Failed to evaluate")
    
    return results