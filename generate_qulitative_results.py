#!/usr/bin/env python3
"""
Comprehensive Qualitative Results Generator for Video Segmentation

This script generates high-quality visualizations for research papers and presentations.
It creates various types of outputs including:
- Frame-by-frame comparisons
- Temporal consistency visualizations
- Error analysis plots
- Performance metrics overlays
- Video outputs for presentations

Usage:
    python generate_qualitative_results.py --config configs/your_config.yaml --checkpoint checkpoints/best_model.pth
"""

import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from datetime import datetime

# Import your project modules
from models.binary_mamba_segmentation import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from utils.evaluation import DAVISEvaluator
from utils.visualization import VideoSegmentationVisualizer

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

class QualitativeResultsGenerator:
    """
    Comprehensive generator for qualitative results suitable for research papers.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'qualitative_results'
    ):
        self.model = model.eval()
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.setup_directories()
        
        # Initialize evaluator and visualizer
        self.evaluator = DAVISEvaluator()
        self.visualizer = VideoSegmentationVisualizer(save_dir=self.output_dir / 'individual_frames')
        
        # Publication settings
        self.dpi = 300
        self.figure_format = 'png'
        self.video_fps = 10
        
        # Color schemes for different visualizations
        self.colors = {
            'prediction': '#2E86AB',      # Blue
            'ground_truth': '#A23B72',    # Magenta
            'overlap': '#F18F01',         # Orange
            'background': '#C73E1D',      # Red
            'success': '#4CAF50',         # Green
            'failure': '#F44336'          # Red
        }
        
        print(f"Qualitative Results Generator initialized")
        print(f"Output directory: {self.output_dir}")
    
    def setup_directories(self):
        """Create organized directory structure for different types of outputs."""
        directories = [
            'individual_frames',
            'sequence_comparisons', 
            'temporal_analysis',
            'error_analysis',
            'performance_plots',
            'videos',
            'paper_figures',
            'presentation_slides'
        ]
        
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def generate_predictions(self, dataloader, max_sequences: int = None) -> Dict:
        """
        Generate predictions for all sequences in the dataloader.
        
        Args:
            dataloader: DataLoader with video sequences
            max_sequences: Maximum number of sequences to process (None for all)
            
        Returns:
            Dictionary containing predictions, ground truth, and metadata
        """
        print("Generating predictions...")
        
        results = {
            'predictions': [],
            'ground_truths': [],
            'sequences': [],
            'frames': [],
            'metrics': []
        }
        
        processed_count = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing sequences")):
            if max_sequences and processed_count >= max_sequences:
                break
            
            # Get data
            frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
            masks = batch['masks'].to(self.device)    # [B, T, H, W]
            sequence = batch.get('sequence', [f"seq_{batch_idx}"])
            
            # Forward pass
            outputs = self.model(frames)
            
            # Use adaptive masks if available, otherwise regular pred_masks
            pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
            
            # Store results (move to CPU to save memory)
            results['predictions'].append(pred_masks[0].cpu())
            results['ground_truths'].append(masks[0].cpu())
            results['sequences'].append(sequence[0] if isinstance(sequence, list) else sequence)
            results['frames'].append(frames[0].cpu())
            
            # Calculate metrics for this sequence
            try:
                seq_metrics = self.evaluator.evaluate_binary_segmentation(
                    predictions=[pred_masks[0].cpu()],
                    ground_truths=[masks[0].cpu()],
                    sequence_names=[sequence[0] if isinstance(sequence, list) else sequence]
                )
                results['metrics'].append(seq_metrics['global'])
            except Exception as e:
                print(f"Warning: Could not calculate metrics for sequence {sequence}: {e}")
                results['metrics'].append({})
            
            processed_count += 1
            
            # Free memory
            del frames, masks, outputs, pred_masks
            torch.cuda.empty_cache()
        
        print(f"Generated predictions for {len(results['sequences'])} sequences")
        return results
    
    def create_paper_figure_comparison(
        self, 
        results: Dict, 
        sequence_indices: List[int] = None,
        frames_per_sequence: int = 4
    ):
        """
        Create publication-quality figure comparing predictions vs ground truth.
        
        Args:
            results: Results dictionary from generate_predictions
            sequence_indices: Which sequences to include (None for automatic selection)
            frames_per_sequence: Number of frames to show per sequence
        """
        print("Creating paper figure comparison...")
        
        # Select sequences automatically if not specified
        if sequence_indices is None:
            # Select sequences with different performance levels
            metrics = results['metrics']
            if metrics and all(m for m in metrics):  # Check if metrics are available
                # Sort by J&F score and select diverse examples
                sorted_indices = sorted(range(len(metrics)), 
                                     key=lambda i: metrics[i].get('J&F', 0))
                
                # Select: worst, median-low, median-high, best
                n_seqs = len(sorted_indices)
                sequence_indices = [
                    sorted_indices[0],                    # Worst
                    sorted_indices[n_seqs//3],           # Low
                    sorted_indices[2*n_seqs//3],         # High
                    sorted_indices[-1]                   # Best
                ]
            else:
                # Fallback to first few sequences
                sequence_indices = list(range(min(4, len(results['sequences']))))
        
        n_sequences = len(sequence_indices)
        
        # Create figure with proper sizing for publication
        fig_width = 4 * frames_per_sequence
        fig_height = 3 * n_sequences
        
        fig, axes = plt.subplots(
            n_sequences, frames_per_sequence * 3,  # 3 columns per frame: input, pred, gt
            figsize=(fig_width, fig_height),
            dpi=self.dpi
        )
        
        if n_sequences == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, seq_num in enumerate(sequence_indices):
            frames = results['frames'][seq_num]
            pred_masks = results['predictions'][seq_num]
            gt_masks = results['ground_truths'][seq_num]
            seq_name = results['sequences'][seq_num]
            
            # Select frames to display
            T = frames.shape[0]
            frame_indices = np.linspace(0, T-1, frames_per_sequence, dtype=int)
            
            for frame_idx, t in enumerate(frame_indices):
                col_base = frame_idx * 3
                
                # Original frame
                frame = frames[t].permute(1, 2, 0).numpy()
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                axes[seq_idx, col_base].imshow(frame)
                axes[seq_idx, col_base].set_title(f'Input\nFrame {t}', fontsize=10)
                axes[seq_idx, col_base].axis('off')
                
                # Prediction overlay
                pred_mask = pred_masks[t, 0].numpy() > 0.5
                pred_vis = frame.copy()
                pred_vis[pred_mask] = pred_vis[pred_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
                
                axes[seq_idx, col_base + 1].imshow(pred_vis)
                axes[seq_idx, col_base + 1].set_title('Prediction', fontsize=10)
                axes[seq_idx, col_base + 1].axis('off')
                
                # Ground truth overlay
                gt_mask = gt_masks[t].numpy() > 0
                gt_vis = frame.copy()
                gt_vis[gt_mask] = gt_vis[gt_mask] * 0.7 + np.array([255, 0, 0]) * 0.3
                
                axes[seq_idx, col_base + 2].imshow(gt_vis)
                axes[seq_idx, col_base + 2].set_title('Ground Truth', fontsize=10)
                axes[seq_idx, col_base + 2].axis('off')
            
            # Add sequence name and metrics on the left
            if results['metrics'][seq_num]:
                metrics = results['metrics'][seq_num]
                metric_text = f"{seq_name}\nJ&F: {metrics.get('J&F', 0):.3f}\nIoU: {metrics.get('iou', 0):.3f}"
            else:
                metric_text = f"{seq_name}"
            
            # Add text annotation
            axes[seq_idx, 0].text(-0.3, 0.5, metric_text, 
                                transform=axes[seq_idx, 0].transAxes,
                                rotation=90, va='center', ha='center',
                                fontsize=9, weight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='green', alpha=0.3, label='Prediction'),
            mpatches.Patch(color='red', alpha=0.3, label='Ground Truth')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / 'paper_figures' / 'qualitative_comparison.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Paper figure saved to: {save_path}")
        return save_path
    
    def create_temporal_consistency_analysis(self, results: Dict, sequence_idx: int = 0):
        """
        Create visualization showing temporal consistency of predictions.
        """
        print("Creating temporal consistency analysis...")
        
        frames = results['frames'][sequence_idx]
        pred_masks = results['predictions'][sequence_idx]
        gt_masks = results['ground_truths'][sequence_idx]
        seq_name = results['sequences'][sequence_idx]
        
        T = frames.shape[0]
        
        # Calculate frame-to-frame changes
        pred_changes = []
        gt_changes = []
        
        for t in range(T - 1):
            # Calculate change between consecutive frames
            pred_curr = (pred_masks[t, 0] > 0.5).float()
            pred_next = (pred_masks[t+1, 0] > 0.5).float()
            pred_change = torch.abs(pred_curr - pred_next).mean().item()
            pred_changes.append(pred_change)
            
            gt_curr = (gt_masks[t] > 0).float()
            gt_next = (gt_masks[t+1] > 0).float()
            gt_change = torch.abs(gt_curr - gt_next).mean().item()
            gt_changes.append(gt_change)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot temporal changes
        frame_nums = list(range(1, T))
        ax1.plot(frame_nums, pred_changes, 'b-', label='Prediction Changes', linewidth=2)
        ax1.plot(frame_nums, gt_changes, 'r--', label='Ground Truth Changes', linewidth=2)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Change Magnitude')
        ax1.set_title(f'Temporal Consistency Analysis - {seq_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create a montage showing key frames
        key_frames = [0, T//4, T//2, 3*T//4, T-1]
        montage_width = len(key_frames)
        
        for i, t in enumerate(key_frames):
            frame = frames[t].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            pred_mask = pred_masks[t, 0].numpy() > 0.5
            
            # Create overlay
            overlay = frame.copy()
            overlay[pred_mask] = overlay[pred_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            # Add to montage subplot
            ax_montage = plt.subplot(2, montage_width, montage_width + i + 1)
            ax_montage.imshow(overlay)
            ax_montage.set_title(f'Frame {t}')
            ax_montage.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / 'temporal_analysis' / f'{seq_name}_temporal_consistency.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_error_analysis_plot(self, results: Dict):
        """
        Create comprehensive error analysis visualization.
        """
        print("Creating error analysis plot...")
        
        if not all(results['metrics']):
            print("Warning: No metrics available for error analysis")
            return None
        
        # Extract metrics
        sequences = results['sequences']
        metrics = results['metrics']
        
        iou_scores = [m.get('iou', 0) for m in metrics]
        f1_scores = [m.get('f1', 0) for m in metrics]
        jf_scores = [m.get('J&F', 0) for m in metrics]
        
        # Create multi-panel error analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Performance distribution
        ax1.hist(jf_scores, bins=20, alpha=0.7, color=self.colors['prediction'])
        ax1.axvline(np.mean(jf_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(jf_scores):.3f}')
        ax1.set_xlabel('J&F Score')
        ax1.set_ylabel('Number of Sequences')
        ax1.set_title('Performance Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. IoU vs F1 scatter
        ax2.scatter(iou_scores, f1_scores, alpha=0.6, c=jf_scores, 
                   cmap='viridis', s=50)
        ax2.set_xlabel('IoU Score')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('IoU vs F1 Performance')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('J&F Score')
        
        # 3. Performance by sequence (sorted)
        sorted_indices = sorted(range(len(jf_scores)), key=lambda i: jf_scores[i])
        sorted_scores = [jf_scores[i] for i in sorted_indices]
        sorted_names = [sequences[i] for i in sorted_indices]
        
        # Show only every nth sequence name to avoid overcrowding
        step = max(1, len(sorted_names) // 10)
        x_pos = range(len(sorted_scores))
        
        ax3.bar(x_pos, sorted_scores, color=self.colors['prediction'], alpha=0.7)
        ax3.set_xlabel('Sequence (sorted by performance)')
        ax3.set_ylabel('J&F Score')
        ax3.set_title('Per-Sequence Performance')
        
        # Add sequence names for worst and best
        if len(sorted_names) > 5:
            ax3.text(0, sorted_scores[0] + 0.01, sorted_names[0], 
                    rotation=90, ha='center', va='bottom', fontsize=8)
            ax3.text(len(sorted_scores)-1, sorted_scores[-1] + 0.01, sorted_names[-1], 
                    rotation=90, ha='center', va='bottom', fontsize=8)
        
        # 4. Error categories
        # Categorize sequences by performance
        excellent = sum(1 for score in jf_scores if score > 0.8)
        good = sum(1 for score in jf_scores if 0.6 < score <= 0.8)
        fair = sum(1 for score in jf_scores if 0.4 < score <= 0.6)
        poor = sum(1 for score in jf_scores if score <= 0.4)
        
        categories = ['Excellent\n(>0.8)', 'Good\n(0.6-0.8)', 'Fair\n(0.4-0.6)', 'Poor\n(≤0.4)']
        counts = [excellent, good, fair, poor]
        colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
        
        ax4.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Performance Categories')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / 'error_analysis' / 'comprehensive_error_analysis.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Save detailed metrics to JSON
        detailed_metrics = {
            'sequences': sequences,
            'metrics': metrics,
            'summary_statistics': {
                'mean_jf': float(np.mean(jf_scores)),
                'std_jf': float(np.std(jf_scores)),
                'mean_iou': float(np.mean(iou_scores)),
                'std_iou': float(np.std(iou_scores)),
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'num_sequences': len(sequences)
            }
        }
        
        json_path = self.output_dir / 'error_analysis' / 'detailed_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        
        print(f"Error analysis saved to: {save_path}")
        return save_path
    
    def create_videos(self, results: Dict, max_videos: int = 5):
        """
        Create video outputs showing predictions over time.
        """
        print("Creating video outputs...")
        
        video_paths = []
        
        for i in range(min(max_videos, len(results['sequences']))):
            seq_name = results['sequences'][i]
            frames = results['frames'][i]
            pred_masks = results['predictions'][i]
            gt_masks = results['ground_truths'][i]
            
            # Create video using the visualizer
            try:
                video_path = self.visualizer.create_video(
                    frames=frames,
                    pred_masks=pred_masks,
                    gt_masks=gt_masks,
                    sequence_name=f"{seq_name}_comparison",
                    fps=self.video_fps
                )
                
                # Move video to our videos directory
                new_path = self.output_dir / 'videos' / f"{seq_name}_comparison.mp4"
                if Path(video_path).exists():
                    Path(video_path).rename(new_path)
                    video_paths.append(new_path)
                    
            except Exception as e:
                print(f"Error creating video for {seq_name}: {e}")
        
        print(f"Created {len(video_paths)} videos")
        return video_paths
    
    def create_presentation_slides(self, results: Dict):
        """
        Create presentation-ready slide images.
        """
        print("Creating presentation slides...")
        
        slides_dir = self.output_dir / 'presentation_slides'
        
        # Slide 1: Method overview with representative example
        if results['sequences']:
            best_idx = 0
            if results['metrics'] and all(results['metrics']):
                # Find best performing sequence
                best_idx = max(range(len(results['metrics'])), 
                             key=lambda i: results['metrics'][i].get('J&F', 0))
            
            self._create_method_overview_slide(results, best_idx, slides_dir)
        
        # Slide 2: Performance comparison
        if len(results['sequences']) >= 3:
            self._create_performance_comparison_slide(results, slides_dir)
        
        # Slide 3: Temporal consistency showcase
        if results['sequences']:
            self._create_temporal_showcase_slide(results, 0, slides_dir)
        
        print(f"Presentation slides saved to: {slides_dir}")
    
    def _create_method_overview_slide(self, results: Dict, seq_idx: int, save_dir: Path):
        """Create a method overview slide."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Video Segmentation Results - Method Overview', fontsize=20, weight='bold')
        
        frames = results['frames'][seq_idx]
        pred_masks = results['predictions'][seq_idx]
        gt_masks = results['ground_truths'][seq_idx]
        seq_name = results['sequences'][seq_idx]
        
        # Show 4 key frames
        T = frames.shape[0]
        frame_indices = [0, T//3, 2*T//3, T-1]
        
        for i, t in enumerate(frame_indices):
            # Original frame
            frame = frames[t].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f'Frame {t+1}', fontsize=14)
            axes[0, i].axis('off')
            
            # Prediction overlay
            pred_mask = pred_masks[t, 0].numpy() > 0.5
            pred_vis = frame.copy()
            pred_vis[pred_mask] = pred_vis[pred_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
            
            axes[1, i].imshow(pred_vis)
            axes[1, i].set_title('Our Prediction', fontsize=14)
            axes[1, i].axis('off')
        
        # Add method labels
        axes[0, 0].text(-0.1, 0.5, 'Input Video', transform=axes[0, 0].transAxes,
                       rotation=90, va='center', ha='center', fontsize=16, weight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Segmentation', transform=axes[1, 0].transAxes,
                       rotation=90, va='center', ha='center', fontsize=16, weight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'slide_1_method_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_performance_comparison_slide(self, results: Dict, save_dir: Path):
        """Create a performance comparison slide."""
        if not all(results['metrics']):
            return
            
        # Select best, median, worst performing sequences
        metrics = results['metrics']
        jf_scores = [m.get('J&F', 0) for m in metrics]
        sorted_indices = sorted(range(len(jf_scores)), key=lambda i: jf_scores[i])
        
        selected_indices = [
            sorted_indices[0],           # Worst
            sorted_indices[len(sorted_indices)//2],  # Median
            sorted_indices[-1]          # Best
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle('Performance Comparison Across Different Scenarios', fontsize=18, weight='bold')
        
        labels = ['Challenging Case', 'Typical Case', 'Best Case']
        
        for row, idx in enumerate(selected_indices):
            frames = results['frames'][idx]
            pred_masks = results['predictions'][idx]
            gt_masks = results['ground_truths'][idx]
            seq_name = results['sequences'][idx]
            metric = results['metrics'][idx]
            
            # Show middle frame
            t = frames.shape[0] // 2
            
            # Original frame
            frame = frames[t].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            axes[row, 0].imshow(frame)
            axes[row, 0].set_title(f'{labels[row]}\nInput', fontsize=12)
            axes[row, 0].axis('off')
            
            # Prediction
            pred_mask = pred_masks[t, 0].numpy() > 0.5
            pred_vis = frame.copy()
            pred_vis[pred_mask] = pred_vis[pred_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            axes[row, 1].imshow(pred_vis)
            axes[row, 1].set_title(f'Prediction\nJ&F: {metric.get("J&F", 0):.3f}', fontsize=12)
            axes[row, 1].axis('off')
            
            # Ground truth
            gt_mask = gt_masks[t].numpy() > 0
            gt_vis = frame.copy()
            gt_vis[gt_mask] = gt_vis[gt_mask] * 0.7 + np.array([255, 0, 0]) * 0.3
            
            axes[row, 2].imshow(gt_vis)
            axes[row, 2].set_title('Ground Truth', fontsize=12)
            axes[row, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'slide_2_performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_showcase_slide(self, results: Dict, seq_idx: int, save_dir: Path):
        """Create a temporal consistency showcase slide."""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Temporal Consistency Demonstration', fontsize=18, weight='bold')
        
        frames = results['frames'][seq_idx]
        pred_masks = results['predictions'][seq_idx]
        
        # Show 5 evenly spaced frames
        T = frames.shape[0]
        frame_indices = np.linspace(0, T-1, 5, dtype=int)
        
        for i, t in enumerate(frame_indices):
            # Original frame
            frame = frames[t].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f'Frame {t+1}', fontsize=12)
            axes[0, i].axis('off')
            
            # Prediction overlay
            pred_mask = pred_masks[t, 0].numpy() > 0.5
            pred_vis = frame.copy()
            pred_vis[pred_mask] = pred_vis[pred_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            axes[1, i].imshow(pred_vis)
            axes[1, i].set_title('Segmentation', fontsize=12)
            axes[1, i].axis('off')
        
        # Add arrow annotations showing temporal flow
        for i in range(4):
            # Add arrow between frames
            axes[0, i].annotate('', xy=(1.1, 0.5), xytext=(0.9, 0.5),
                              xycoords='axes fraction', textcoords='axes fraction',
                              arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'slide_3_temporal_consistency.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_all_results(self, dataloader, max_sequences: int = None) -> Dict:
        """
        Generate comprehensive qualitative results.
        
        Args:
            dataloader: DataLoader with validation/test data
            max_sequences: Maximum number of sequences to process
            
        Returns:
            Dictionary with paths to all generated visualizations
        """
        print("=== Starting Comprehensive Qualitative Results Generation ===")
        
        # Step 1: Generate predictions
        results = self.generate_predictions(dataloader, max_sequences)
        
        if not results['sequences']:
            print("No sequences processed. Exiting.")
            return {}
        
        # Step 2: Create paper-quality figures
        paper_figure_path = self.create_paper_figure_comparison(results)
        
        # Step 3: Create error analysis
        error_analysis_path = self.create_error_analysis_plot(results)
        
        # Step 4: Create temporal consistency analysis for first few sequences
        temporal_paths = []
        for i in range(min(3, len(results['sequences']))):
            temporal_path = self.create_temporal_consistency_analysis(results, i)
            temporal_paths.append(temporal_path)
        
        # Step 5: Create videos
        video_paths = self.create_videos(results, max_videos=5)
        
        # Step 6: Create presentation slides
        self.create_presentation_slides(results)
        
        # Step 7: Generate individual sequence visualizations
        individual_paths = self._generate_individual_visualizations(results)
        
        # Step 8: Create summary report
        summary_path = self._create_summary_report(results)
        
        # Compile all paths
        generated_files = {
            'paper_figure': paper_figure_path,
            'error_analysis': error_analysis_path,
            'temporal_analysis': temporal_paths,
            'videos': video_paths,
            'individual_visualizations': individual_paths,
            'summary_report': summary_path,
            'presentation_slides': list((self.output_dir / 'presentation_slides').glob('*.png')),
            'output_directory': str(self.output_dir)
        }
        
        print("=== Qualitative Results Generation Complete ===")
        print(f"Results saved to: {self.output_dir}")
        print(f"Generated {len(results['sequences'])} sequence analyses")
        
        return generated_files
    
    def _generate_individual_visualizations(self, results: Dict) -> List[Path]:
        """Generate detailed visualizations for each sequence."""
        print("Generating individual sequence visualizations...")
        
        individual_paths = []
        
        for i, seq_name in enumerate(results['sequences']):
            frames = results['frames'][i]
            pred_masks = results['predictions'][i]
            gt_masks = results['ground_truths'][i]
            
            # Use the visualizer to create sequence analysis
            figures = self.visualizer.visualize_sequence(
                frames=frames,
                pred_masks=pred_masks,
                gt_masks=gt_masks,
                sequence_name=seq_name,
                max_frames=8
            )
            
            # Create dashboard for this sequence
            if results['metrics'][i]:
                dashboard_fig = self.visualizer.create_analysis_dashboard(
                    frames=frames,
                    pred_masks=pred_masks,
                    gt_masks=gt_masks,
                    metrics=results['metrics'][i],
                    sequence_name=seq_name,
                    save=True
                )
                plt.close(dashboard_fig)
            
            individual_paths.extend([
                self.output_dir / 'individual_frames' / f"{seq_name}_frame_{t:03d}.png"
                for t in range(min(8, frames.shape[0]))
            ])
        
        return individual_paths
    
    def _create_summary_report(self, results: Dict) -> Path:
        """Create a comprehensive summary report."""
        print("Creating summary report...")
        
        report_path = self.output_dir / 'summary_report.html'
        
        # Calculate overall statistics
        if results['metrics'] and all(results['metrics']):
            metrics = results['metrics']
            jf_scores = [m.get('J&F', 0) for m in metrics]
            iou_scores = [m.get('iou', 0) for m in metrics]
            f1_scores = [m.get('f1', 0) for m in metrics]
            
            stats = {
                'num_sequences': len(results['sequences']),
                'mean_jf': np.mean(jf_scores),
                'std_jf': np.std(jf_scores),
                'mean_iou': np.mean(iou_scores),
                'mean_f1': np.mean(f1_scores),
                'best_sequence': results['sequences'][np.argmax(jf_scores)],
                'worst_sequence': results['sequences'][np.argmin(jf_scores)],
                'best_jf': max(jf_scores),
                'worst_jf': min(jf_scores)
            }
        else:
            stats = {
                'num_sequences': len(results['sequences']),
                'mean_jf': 'N/A',
                'std_jf': 'N/A',
                'mean_iou': 'N/A',
                'mean_f1': 'N/A',
                'best_sequence': 'N/A',
                'worst_sequence': 'N/A',
                'best_jf': 'N/A',
                'worst_jf': 'N/A'
            }
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Segmentation Qualitative Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
                .file-list {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .sequence-list {{ max-height: 300px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Video Segmentation Qualitative Results</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Model Configuration: {self.config.get('model', {}).get('hidden_dims', 'N/A')}</p>
            </div>
            
            <h2>Performance Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>{stats['num_sequences']}</h3>
                    <p>Total Sequences</p>
                </div>
                <div class="stat-box">
                    <h3>{stats['mean_jf']:.3f}</h3>
                    <p>Mean J&F Score</p>
                </div>
                <div class="stat-box">
                    <h3>{stats['mean_iou']:.3f}</h3>
                    <p>Mean IoU</p>
                </div>
                <div class="stat-box">
                    <h3>{stats['best_jf']:.3f}</h3>
                    <p>Best J&F Score</p>
                </div>
            </div>
            
            <h2>Generated Files</h2>
            <div class="file-list">
                <h3>Paper Figures</h3>
                <ul>
                    <li>📊 <a href="paper_figures/qualitative_comparison.png">Main Qualitative Comparison</a></li>
                    <li>📈 <a href="error_analysis/comprehensive_error_analysis.png">Error Analysis</a></li>
                </ul>
                
                <h3>Presentation Materials</h3>
                <ul>
                    <li>🎥 Videos in <code>videos/</code> directory</li>
                    <li>📑 Presentation slides in <code>presentation_slides/</code> directory</li>
                </ul>
                
                <h3>Detailed Analysis</h3>
                <ul>
                    <li>🔍 Individual frame visualizations in <code>individual_frames/</code></li>
                    <li>⏱️ Temporal analysis in <code>temporal_analysis/</code></li>
                </ul>
            </div>
            
            <h2>Sequence List</h2>
            <div class="sequence-list">
                <table border="1" style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th>Sequence Name</th>
                        <th>J&F Score</th>
                        <th>IoU</th>
                        <th>F1 Score</th>
                    </tr>
        """
        
        # Add sequence rows
        for i, seq_name in enumerate(results['sequences']):
            if results['metrics'][i]:
                m = results['metrics'][i]
                html_content += f"""
                    <tr>
                        <td>{seq_name}</td>
                        <td>{m.get('J&F', 0):.3f}</td>
                        <td>{m.get('iou', 0):.3f}</td>
                        <td>{m.get('f1', 0):.3f}</td>
                    </tr>
                """
            else:
                html_content += f"""
                    <tr>
                        <td>{seq_name}</td>
                        <td>N/A</td>
                        <td>N/A</td>
                        <td>N/A</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <h2>Usage Instructions</h2>
            <div class="file-list">
                <h3>For Research Papers:</h3>
                <ul>
                    <li>Use <code>paper_figures/qualitative_comparison.png</code> for main results figure</li>
                    <li>Use <code>error_analysis/comprehensive_error_analysis.png</code> for performance analysis</li>
                    <li>PDF versions are available for high-quality publication</li>
                </ul>
                
                <h3>For Presentations:</h3>
                <ul>
                    <li>Use slides from <code>presentation_slides/</code> directory</li>
                    <li>Use videos from <code>videos/</code> for dynamic demonstrations</li>
                </ul>
                
                <h3>For Detailed Analysis:</h3>
                <ul>
                    <li>Check individual sequence dashboards for specific cases</li>
                    <li>Use temporal analysis for understanding model behavior over time</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path


def main():
    """Main function to generate qualitative results."""
    parser = argparse.ArgumentParser(description='Generate qualitative results for video segmentation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='qualitative_results',
                       help='Output directory for results')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to process')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--specific-sequence', type=str, default=None,
                       help='Process only this specific sequence')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = build_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    
    # Create data transform (no augmentation for evaluation)
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        normalize=True,
        train=False
    )
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split=args.split,
        batch_size=1,  # Process one sequence at a time
        transform=transform,
        specific_sequence=args.specific_sequence,
        **{k: v for k, v in config['dataset'].items() 
           if k not in ['batch_size', 'augmentation']}
    )
    
    print(f"Created dataloader with {len(dataloader)} sequences")
    
    # Initialize results generator
    generator = QualitativeResultsGenerator(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    
    # Generate all results
    generated_files = generator.generate_all_results(
        dataloader=dataloader,
        max_sequences=args.max_sequences
    )
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITATIVE RESULTS GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {generated_files['output_directory']}")
    print(f"Summary report: {generated_files['summary_report']}")
    print("\nKey files for papers/presentations:")
    print(f"  📊 Main figure: {generated_files['paper_figure']}")
    if generated_files['error_analysis']:
        print(f"  📈 Error analysis: {generated_files['error_analysis']}")
    print(f"  🎥 Videos: {len(generated_files['videos'])} created")
    print(f"  📑 Slides: {len(generated_files['presentation_slides'])} created")
    print("\nOpen the summary report (HTML file) for detailed navigation!")


if __name__ == '__main__':
    main()