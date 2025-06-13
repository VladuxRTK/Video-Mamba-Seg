#!/usr/bin/env python3
"""
Usage examples and practical scripts for temporal stability analysis
with your VideoMamba model on DAVIS dataset.
"""

import torch
import yaml
import argparse
from pathlib import Path

# Example 1: Basic temporal stability analysis for a single sequence
def example_single_sequence_analysis():
    """
    Example of analyzing temporal stability for a single DAVIS sequence.
    This shows the basic workflow and what to expect from the analysis.
    """
    print("Example 1: Single Sequence Temporal Stability Analysis")
    print("=" * 50)
    
    # Your model and config paths
    model_path = "checkpoints/emergency_fix/best_model.pth"
    config_path = "configs/emergency_fix.yaml"
    sequence_name = "blackswan"  # Example DAVIS sequence
    
    # Run the analysis
    from temporal_stability_davis import evaluate_temporal_stability_on_davis
    
    results = evaluate_temporal_stability_on_davis(
        model_path=model_path,
        config_path=config_path,
        sequence_name=sequence_name,
        save_dir=f"temporal_analysis_{sequence_name}"
    )
    
    # The results will include:
    # - T_mean: Overall temporal stability score (0-1, higher is better)
    # - T_recall: Percentage of frame transitions with high stability
    # - T_decay: How much stability degrades over time
    # - Visualizations saved to the specified directory
    
    print(f"Results for {sequence_name}:")
    for key, value in results.items():
        if key != 'frame_to_frame_stabilities':
            print(f"  {key}: {value:.4f}")


# Example 2: Comprehensive evaluation with temporal stability
def example_comprehensive_evaluation():
    """
    Example of running comprehensive evaluation on the entire validation set
    including both DAVIS metrics and temporal stability.
    """
    print("\nExample 2: Comprehensive Evaluation")
    print("=" * 50)
    
    model_path = "checkpoints/emergency_fix/best_model.pth"
    config_path = "configs/emergency_fix.yaml"
    
    from enhanced_evaluation_integration import evaluate_videomamba_comprehensive
    
    results = evaluate_videomamba_comprehensive(
        model_path=model_path,
        config_path=config_path,
        dataset_split='val',
        save_visualizations=True
    )
    
    # Results structure:
    # {
    #   'davis_metrics': {'J&F': ..., 'J_mean': ..., 'F_mean': ...},
    #   'temporal_metrics': {'global_T_mean': ..., 'global_T_recall': ...},
    #   'per_sequence_davis': {...},
    #   'per_sequence_temporal': {...}
    # }
    
    return results


# Example 3: Integration with your existing training loop
def example_training_integration():
    """
    Example showing how to integrate temporal stability monitoring
    into your existing training pipeline.
    """
    print("\nExample 3: Training Integration")
    print("=" * 50)
    
    # Add this to your training script (train.py) or Trainer class
    
    # In your validation loop, add temporal stability tracking:
    example_code = """
    # Add to your Trainer.validate() method:
    
    def validate_with_temporal_analysis(self, val_loader):
        standard_metrics = self.validate(val_loader)  # Your existing validation
        
        # Add temporal stability analysis for a few sequences
        from temporal_stability_davis import TemporalStabilityAnalyzer
        analyzer = TemporalStabilityAnalyzer()
        
        temporal_scores = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 3:  # Analyze first 3 sequences only
                    break
                
                frames = batch['frames'].to(self.device)
                gt_masks = batch['masks'].to(self.device)
                
                outputs = self.model(frames)
                pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                
                # Compute temporal stability
                temp_metrics = analyzer.compute_temporal_stability(
                    pred_masks[0].cpu().squeeze(),
                    gt_masks[0].cpu()
                )
                
                temporal_scores.append(temp_metrics['T_mean'])
        
        # Add average temporal stability to your metrics
        if temporal_scores:
            avg_temporal = np.mean(temporal_scores)
            standard_metrics['temporal_stability'] = avg_temporal
            self.logger.info(f"Temporal Stability: {avg_temporal:.4f}")
        
        return standard_metrics
    """
    
    print("Integration code example:")
    print(example_code)


# Example 4: Comparing different model configurations
def example_model_comparison():
    """
    Example of comparing temporal stability across different model configurations
    or training epochs to understand how temporal consistency evolves.
    """
    # print("\nExample 4: Model Comparison")
    # print("=" * 50)
    
    # # Compare different checkpoints
    # model_configs = [
    #     ("checkpoints/model_epoch_10.pth", "Epoch 10"),
    #     ("checkpoints/model_epoch_50.pth", "Epoch 50"),
    #     ("checkpoints/model_best.pth", "Best Model"),
    # ]
    
    # model_path = "checkpoints/emergency_fix/best_model.pth"
    # config_path = "configs/emergency_fix.yaml"
    # test_sequence = "blackswan"
    
    # from enhanced_evaluation_integration import compare_temporal_stability
    
    # model_paths = [config[0] for config in model_configs]
    # model_names = [config[1] for config in model_configs]
    
    # comparison_results = compare_temporal_stability(
    #     model_paths=model_paths,
    #     model_names=model_names,
    #     config_path=config_path,
    #     sequence_name=test_sequence
    # )
    
    # # This will show you how temporal stability improves (or degrades) during training
    # print("Comparison completed. Check the output for temporal stability trends.")


# Example 5: Analyzing your VideoMamba's strengths and weaknesses
def example_detailed_analysis():
    """
    Example of detailed analysis to understand where your VideoMamba model
    excels and where it struggles with temporal consistency.
    """
    print("\nExample 5: Detailed Temporal Analysis")
    print("=" * 50)
    
    # Analyze specific challenging sequences
    challenging_sequences = [
        "blackswan",      # Fast motion, deformation
        "bmx-trees",      # Camera motion, occlusion
        "breakdance",     # Complex motion, multiple objects
        "camel",          # Slow motion, stable
        "car-roundabout", # Predictable motion
    ]
    
    model_path = "checkpoints/emergency_fix/best_model.pth"
    config_path = "configs/emergency_fix.yaml"
    
    sequence_analysis = {}
    
    for sequence in challenging_sequences:
        print(f"\nAnalyzing {sequence}...")
        
        from enhanced_evaluation_integration import quick_temporal_analysis
        
        try:
            results = quick_temporal_analysis(
                model_path=model_path,
                config_path=config_path,
                sequence_name=sequence,
                save_dir=f"detailed_analysis_{sequence}"
            )
            
            sequence_analysis[sequence] = results
            
        except Exception as e:
            print(f"Error analyzing {sequence}: {e}")
            sequence_analysis[sequence] = None
    
    # Analyze patterns
    print("\n" + "=" * 60)
    print("TEMPORAL STABILITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    stable_sequences = []
    unstable_sequences = []
    
    for seq, results in sequence_analysis.items():
        if results and 'T_mean' in results:
            t_score = results['T_mean']
            print(f"{seq:15s}: T-score = {t_score:.4f}")
            
            if t_score > 0.90:
                stable_sequences.append(seq)
            elif t_score < 0.80:
                unstable_sequences.append(seq)
    
    print(f"\nStable sequences (T > 0.90): {stable_sequences}")
    print(f"Unstable sequences (T < 0.80): {unstable_sequences}")
    
    # Insights based on your VideoMamba results from the paper
    print("\nInsights for VideoMamba:")
    print("- VideoMamba excels at temporal consistency (T=0.974 reported)")
    print("- Struggles with spatial boundary precision (F=0.169)")
    print("- 144x parameter reduction while maintaining temporal performance")
    
    return sequence_analysis


# Main script for command-line usage
def main():
    parser = argparse.ArgumentParser(description='VideoMamba Temporal Stability Analysis')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'comprehensive', 'compare', 'detailed'],
                       help='Analysis mode to run')
    parser.add_argument('--model', type=str, default='checkpoints/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/mamba_binary_efficient.yaml',
                       help='Path to config file')
    parser.add_argument('--sequence', type=str, default='blackswan',
                       help='Sequence name for single sequence analysis')
    parser.add_argument('--output-dir', type=str, default='temporal_stability_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single sequence analysis
        from temporal_stability_davis import evaluate_temporal_stability_on_davis
        
        results = evaluate_temporal_stability_on_davis(
            model_path=args.model,
            config_path=args.config,
            sequence_name=args.sequence,
            save_dir=args.output_dir
        )
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    elif args.mode == 'comprehensive':
        # Comprehensive evaluation
        from enhanced_evaluation_integration import evaluate_videomamba_comprehensive
        
        results = evaluate_videomamba_comprehensive(
            model_path=args.model,
            config_path=args.config,
            save_visualizations=True
        )
        
        print("Comprehensive evaluation complete.")
        
    elif args.mode == 'compare':
        print("Comparison mode requires multiple model paths.")
        print("Edit the script to specify the models you want to compare.")
        
    elif args.mode == 'detailed':
        # Detailed analysis on multiple sequences
        example_detailed_analysis()


# Expected results interpretation guide
def interpretation_guide():
    """
    Guide for interpreting temporal stability results in the context of your VideoMamba model.
    """
    guide_text = """
    TEMPORAL STABILITY INTERPRETATION GUIDE
    =====================================
    
    Based on your VideoMamba paper results (T=0.974), here's what to expect:
    
    T-Score Ranges:
    - T > 0.95: Excellent temporal stability (VideoMamba target range)
    - T > 0.90: Good temporal stability
    - T > 0.80: Moderate temporal stability
    - T < 0.80: Poor temporal stability
    
    VideoMamba Strengths (from your analysis):
    ✓ Exceptional temporal consistency (T=0.974)
    ✓ 144x parameter reduction vs transformers
    ✓ Linear complexity for long sequences
    ✓ 18.5 FPS inference speed
    
    VideoMamba Limitations (from your analysis):
    ✗ Spatial boundary precision (F=0.169)
    ✗ Struggles with pixel-perfect accuracy
    ✗ Limited spatial reasoning capability
    
    What the visualizations will show:
    1. Frame-to-frame stability plot: Should show consistently high values (>0.95)
    2. Temporal consistency video: Minimal flickering, stable object boundaries
    3. Change detection: Small, smooth changes between frames
    4. Stability distribution: Concentrated around high values
    
    Use cases where VideoMamba excels:
    - Object tracking applications
    - Real-time video processing
    - Resource-constrained environments
    - Applications prioritizing temporal consistency over spatial precision
    """
    
    print(guide_text)


if __name__ == "__main__":
    print("VideoMamba Temporal Stability Analysis Examples")
    print("=" * 50)
    print("Choose an example to run:")
    print("1. Single sequence analysis")
    print("2. Comprehensive evaluation")
    print("3. Training integration example")
    print("4. Model comparison")
    print("5. Detailed analysis")
    print("6. Interpretation guide")
    print("7. Run command-line interface")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        example_single_sequence_analysis()
    elif choice == '2':
        example_comprehensive_evaluation()
    elif choice == '3':
        example_training_integration()
    elif choice == '4':
        example_model_comparison()
    elif choice == '5':
        example_detailed_analysis()
    elif choice == '6':
        interpretation_guide()
    elif choice == '7':
        main()
    else:
        print("Invalid choice. Please run the script again.")
