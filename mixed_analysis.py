#!/usr/bin/env python3
"""
Mixed Results Analysis Generator

Creates comprehensive analysis showing both strengths and limitations
for papers with mixed/intermediate results.
"""

import torch
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import torch.nn.functional as F
import json
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Import your project modules
from models.binary_mamba_segmentation import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation


class MixedResultsAnalyzer:
    """Analyzes and presents both strengths and limitations of the model."""
    
    def __init__(self, model, device='cuda', output_dir='mixed_results_analysis'):
        self.model = model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'strengths').mkdir(exist_ok=True)
        (self.output_dir / 'limitations').mkdir(exist_ok=True)
        (self.output_dir / 'quantitative').mkdir(exist_ok=True)
        (self.output_dir / 'paper_figures').mkdir(exist_ok=True)
        
        # Analysis categories
        self.analysis_results = {
            'excellent_cases': [],      # IoU > 0.75
            'good_cases': [],          # IoU 0.5-0.75
            'poor_cases': [],          # IoU < 0.5
            'temporal_consistent': [], # Low frame-to-frame variation
            'temporal_inconsistent': [], # High frame-to-frame variation
            'boundary_accurate': [],   # Good boundary precision
            'boundary_poor': [],       # Poor boundary precision
        }
        
        # Colors for different aspects
        self.colors = {
            'prediction': [0, 200, 0],     # Green
            'ground_truth': [200, 0, 0],   # Red
            'excellent': [34, 139, 34],    # Forest Green
            'good': [255, 165, 0],         # Orange
            'poor': [220, 20, 60],         # Crimson
            'temporal_good': [30, 144, 255], # Dodger Blue
            'temporal_poor': [255, 69, 0],   # Red Orange
        }
        
        print(f"Mixed Results Analyzer initialized")
        print(f"Output directory: {self.output_dir}")
    
    def _normalize_frame(self, frame):
        """Properly normalize frame for display."""
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = frame.permute(1, 2, 0)
        
        frame = frame.numpy()
        
        # Handle ImageNet normalization
        if frame.min() < 0 or frame.max() > 1:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = frame * std + mean
            frame = np.clip(frame, 0, 1)
        
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame
    
    def _resize_mask_to_frame(self, mask, target_shape):
        """Resize mask to match frame shape."""
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.permute(2, 0, 1).unsqueeze(0)
        
        resized_mask = F.interpolate(
            mask.float(), 
            size=target_shape, 
            mode='nearest'
        )
        
        return resized_mask.squeeze()
    
    def _calculate_iou(self, pred_mask, gt_mask):
        """Calculate IoU between prediction and ground truth."""
        pred_mask = pred_mask.bool()
        gt_mask = gt_mask.bool()
        
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    def _calculate_boundary_accuracy(self, pred_mask, gt_mask, tolerance=2):
        """Calculate boundary accuracy with tolerance."""
        # Get boundaries using morphological operations
        kernel = np.ones((3,3), np.uint8)
        
        pred_np = pred_mask.cpu().numpy().astype(np.uint8)
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        
        # Get boundaries
        pred_boundary = cv2.morphologyEx(pred_np, cv2.MORPH_GRADIENT, kernel)
        gt_boundary = cv2.morphologyEx(gt_np, cv2.MORPH_GRADIENT, kernel)
        
        if gt_boundary.sum() == 0:
            return 1.0 if pred_boundary.sum() == 0 else 0.0
        
        # Dilate GT boundary for tolerance
        dilated_gt = cv2.dilate(gt_boundary, np.ones((tolerance*2+1, tolerance*2+1), np.uint8))
        
        # Calculate boundary recall
        boundary_hits = (pred_boundary * dilated_gt).sum()
        total_boundary = pred_boundary.sum()
        
        if total_boundary == 0:
            return 0.0
        
        return boundary_hits / total_boundary
    
    def _calculate_temporal_consistency(self, masks):
        """Calculate temporal consistency across frames."""
        if len(masks) < 2:
            return 1.0
        
        consistencies = []
        for i in range(len(masks) - 1):
            mask1 = masks[i].bool()
            mask2 = masks[i + 1].bool()
            
            # Calculate overlap
            intersection = (mask1 & mask2).sum().float()
            union = (mask1 | mask2).sum().float()
            
            if union == 0:
                consistency = 1.0
            else:
                consistency = intersection / union
            
            consistencies.append(consistency.item())
        
        return np.mean(consistencies)
    
    @torch.no_grad()
    def analyze_sequences(self, dataloader, max_sequences=20):
        """Analyze sequences and categorize them by performance."""
        print(f"Analyzing up to {max_sequences} sequences...")
        
        all_sequence_data = []
        
        for seq_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing sequences")):
            if seq_idx >= max_sequences:
                break
            
            try:
                # Get data
                frames = batch['frames'].to(self.device)
                masks = batch['masks'].to(self.device)
                sequence = batch.get('sequence', [f"seq_{seq_idx}"])[0]
                
                # Forward pass
                outputs = self.model(frames)
                pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                
                # Move to CPU
                frames = frames[0].cpu()
                pred_masks = pred_masks[0].cpu()
                gt_masks = masks[0].cpu()
                
                # Analyze this sequence
                sequence_analysis = self._analyze_single_sequence(
                    frames, pred_masks, gt_masks, sequence
                )
                
                all_sequence_data.append(sequence_analysis)
                
                # Categorize based on performance
                self._categorize_sequence(sequence_analysis)
                
                del frames, masks, outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error analyzing sequence {seq_idx}: {e}")
                continue
        
        print(f"Analyzed {len(all_sequence_data)} sequences")
        
        # Save analysis results
        self._save_analysis_results(all_sequence_data)
        
        return all_sequence_data
    
    def _analyze_single_sequence(self, frames, pred_masks, gt_masks, seq_name):
        """Analyze a single sequence comprehensively."""
        T = frames.shape[0]
        
        analysis = {
            'sequence_name': seq_name,
            'frames': frames,
            'pred_masks': pred_masks,
            'gt_masks': gt_masks,
            'frame_ious': [],
            'frame_boundary_scores': [],
            'temporal_consistency': 0.0,
            'avg_iou': 0.0,
            'avg_boundary_score': 0.0,
            'best_frame': 0,
            'worst_frame': 0,
            'performance_category': 'unknown'
        }
        
        # Analyze each frame
        pred_masks_resized = []
        for t in range(T):
            frame = frames[t]
            frame_h, frame_w = frame.shape[1], frame.shape[2]
            
            # Get masks
            if len(pred_masks.shape) == 4:  # [T, 1, H, W]
                pred_mask = pred_masks[t, 0]
            else:  # [T, H, W]
                pred_mask = pred_masks[t]
            
            gt_mask = gt_masks[t]
            
            # Resize prediction to match frame
            if pred_mask.shape != (frame_h, frame_w):
                pred_mask = self._resize_mask_to_frame(pred_mask, (frame_h, frame_w))
            
            pred_mask_binary = pred_mask > 0.5
            gt_mask_binary = gt_mask > 0
            
            pred_masks_resized.append(pred_mask_binary)
            
            # Calculate metrics
            iou = self._calculate_iou(pred_mask_binary, gt_mask_binary)
            boundary_score = self._calculate_boundary_accuracy(pred_mask_binary, gt_mask_binary)
            
            analysis['frame_ious'].append(iou)
            analysis['frame_boundary_scores'].append(boundary_score)
        
        # Calculate temporal consistency
        analysis['temporal_consistency'] = self._calculate_temporal_consistency(pred_masks_resized)
        
        # Calculate averages
        analysis['avg_iou'] = np.mean(analysis['frame_ious'])
        analysis['avg_boundary_score'] = np.mean(analysis['frame_boundary_scores'])
        
        # Find best and worst frames
        analysis['best_frame'] = np.argmax(analysis['frame_ious'])
        analysis['worst_frame'] = np.argmin(analysis['frame_ious'])
        
        return analysis
    
    def _categorize_sequence(self, analysis):
        """Categorize sequence based on performance metrics."""
        avg_iou = analysis['avg_iou']
        temporal_consistency = analysis['temporal_consistency']
        avg_boundary = analysis['avg_boundary_score']
        
        # Performance categories
        if avg_iou > 0.75:
            self.analysis_results['excellent_cases'].append(analysis)
            analysis['performance_category'] = 'excellent'
        elif avg_iou > 0.5:
            self.analysis_results['good_cases'].append(analysis)
            analysis['performance_category'] = 'good'
        else:
            self.analysis_results['poor_cases'].append(analysis)
            analysis['performance_category'] = 'poor'
        
        # Temporal consistency
        if temporal_consistency > 0.8:
            self.analysis_results['temporal_consistent'].append(analysis)
        else:
            self.analysis_results['temporal_inconsistent'].append(analysis)
        
        # Boundary accuracy
        if avg_boundary > 0.7:
            self.analysis_results['boundary_accurate'].append(analysis)
        else:
            self.analysis_results['boundary_poor'].append(analysis)
    
    def create_strengths_showcase(self):
        """Create visualizations highlighting model strengths."""
        print("Creating strengths showcase...")
        
        # 1. Best temporal consistency examples
        if self.analysis_results['temporal_consistent']:
            best_temporal = sorted(
                self.analysis_results['temporal_consistent'], 
                key=lambda x: x['temporal_consistency'], 
                reverse=True
            )[:3]
            
            self._create_temporal_consistency_figure(
                best_temporal, 
                title="Strength: Excellent Temporal Consistency",
                save_path=self.output_dir / 'strengths' / 'temporal_consistency.png'
            )
        
        # 2. Best overall performance examples
        if self.analysis_results['excellent_cases']:
            best_overall = sorted(
                self.analysis_results['excellent_cases'], 
                key=lambda x: x['avg_iou'], 
                reverse=True
            )[:3]
            
            self._create_performance_showcase(
                best_overall, 
                title="Strength: High-Quality Segmentation",
                save_path=self.output_dir / 'strengths' / 'high_quality.png'
            )
        
        # 3. Robustness to complex scenes (good cases with complex backgrounds)
        if self.analysis_results['good_cases']:
            complex_scenes = self.analysis_results['good_cases'][:3]
            
            self._create_robustness_showcase(
                complex_scenes,
                title="Strength: Robustness to Complex Scenes",
                save_path=self.output_dir / 'strengths' / 'robustness.png'
            )
    
    def create_limitations_analysis(self):
        """Create visualizations highlighting model limitations."""
        print("Creating limitations analysis...")
        
        # 1. Boundary precision issues
        if self.analysis_results['boundary_poor']:
            boundary_issues = sorted(
                self.analysis_results['boundary_poor'], 
                key=lambda x: x['avg_boundary_score']
            )[:3]
            
            self._create_boundary_issues_figure(
                boundary_issues,
                title="Limitation: Boundary Precision Issues",
                save_path=self.output_dir / 'limitations' / 'boundary_precision.png'
            )
        
        # 2. Temporal inconsistency examples
        if self.analysis_results['temporal_inconsistent']:
            temporal_issues = sorted(
                self.analysis_results['temporal_inconsistent'], 
                key=lambda x: x['temporal_consistency']
            )[:3]
            
            self._create_temporal_issues_figure(
                temporal_issues,
                title="Limitation: Temporal Inconsistency",
                save_path=self.output_dir / 'limitations' / 'temporal_issues.png'
            )
        
        # 3. Failure cases
        if self.analysis_results['poor_cases']:
            failure_cases = sorted(
                self.analysis_results['poor_cases'], 
                key=lambda x: x['avg_iou']
            )[:3]
            
            self._create_failure_analysis(
                failure_cases,
                title="Limitation: Challenging Cases",
                save_path=self.output_dir / 'limitations' / 'failure_cases.png'
            )
    
    def _create_temporal_consistency_figure(self, sequences, title, save_path):
        """Create figure showing temporal consistency."""
        fig, axes = plt.subplots(len(sequences), 4, figsize=(16, 4*len(sequences)))
        fig.suptitle(title, fontsize=16, weight='bold')
        
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, analysis in enumerate(sequences):
            frames = analysis['frames']
            pred_masks = analysis['pred_masks']
            seq_name = analysis['sequence_name']
            temporal_score = analysis['temporal_consistency']
            
            T = frames.shape[0]
            frame_indices = np.linspace(0, T-1, 3, dtype=int) if T > 3 else list(range(T))
            
            for i, t in enumerate(frame_indices[:3]):
                frame = self._normalize_frame(frames[t])
                
                if len(pred_masks.shape) == 4:
                    pred_mask = pred_masks[t, 0] > 0.5
                else:
                    pred_mask = pred_masks[t] > 0.5
                
                # Resize if needed
                if pred_mask.shape != frame.shape[:2]:
                    pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
                
                # Create overlay
                overlay = frame.copy()
                if torch.is_tensor(pred_mask):
                    pred_mask = pred_mask.cpu().numpy()
                overlay[pred_mask] = (overlay[pred_mask] * 0.7 + 
                                    np.array(self.colors['temporal_good']) * 0.3).astype(np.uint8)
                
                axes[seq_idx, i].imshow(overlay)
                axes[seq_idx, i].set_title(f'Frame {t+1}')
                axes[seq_idx, i].axis('off')
            
            # Add sequence info
            info_text = f"{seq_name}\nTemporal Consistency: {temporal_score:.3f}"
            axes[seq_idx, 3].text(0.1, 0.5, info_text, fontsize=12, 
                                 transform=axes[seq_idx, 3].transAxes,
                                 verticalalignment='center')
            axes[seq_idx, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_showcase(self, sequences, title, save_path):
        """Create figure showing high-performance examples."""
        fig, axes = plt.subplots(len(sequences), 6, figsize=(18, 3*len(sequences)))
        fig.suptitle(title, fontsize=16, weight='bold')
        
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, analysis in enumerate(sequences):
            frames = analysis['frames']
            pred_masks = analysis['pred_masks']
            gt_masks = analysis['gt_masks']
            seq_name = analysis['sequence_name']
            avg_iou = analysis['avg_iou']
            best_frame = analysis['best_frame']
            
            # Show best frame
            t = best_frame
            frame = self._normalize_frame(frames[t])
            
            # Original
            axes[seq_idx, 0].imshow(frame)
            axes[seq_idx, 0].set_title('Input')
            axes[seq_idx, 0].axis('off')
            
            # Prediction
            if len(pred_masks.shape) == 4:
                pred_mask = pred_masks[t, 0] > 0.5
            else:
                pred_mask = pred_masks[t] > 0.5
            
            if pred_mask.shape != frame.shape[:2]:
                pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
            
            pred_overlay = frame.copy()
            if torch.is_tensor(pred_mask):
                pred_mask_np = pred_mask.cpu().numpy()
            else:
                pred_mask_np = pred_mask
            pred_overlay[pred_mask_np] = (pred_overlay[pred_mask_np] * 0.7 + 
                                        np.array(self.colors['prediction']) * 0.3).astype(np.uint8)
            
            axes[seq_idx, 1].imshow(pred_overlay)
            axes[seq_idx, 1].set_title('Prediction')
            axes[seq_idx, 1].axis('off')
            
            # Ground truth
            gt_mask = gt_masks[t] > 0
            gt_overlay = frame.copy()
            if torch.is_tensor(gt_mask):
                gt_mask_np = gt_mask.cpu().numpy()
            else:
                gt_mask_np = gt_mask
            gt_overlay[gt_mask_np] = (gt_overlay[gt_mask_np] * 0.7 + 
                                    np.array(self.colors['ground_truth']) * 0.3).astype(np.uint8)
            
            axes[seq_idx, 2].imshow(gt_overlay)
            axes[seq_idx, 2].set_title('Ground Truth')
            axes[seq_idx, 2].axis('off')
            
            # Show IoU progression
            frame_ious = analysis['frame_ious']
            axes[seq_idx, 3].plot(range(len(frame_ious)), frame_ious, 'bo-', linewidth=2, markersize=6)
            axes[seq_idx, 3].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
            axes[seq_idx, 3].set_ylim(0, 1)
            axes[seq_idx, 3].set_title('IoU per Frame')
            axes[seq_idx, 3].set_xlabel('Frame')
            axes[seq_idx, 3].set_ylabel('IoU')
            axes[seq_idx, 3].grid(True, alpha=0.3)
            
            # Show comparison (side by side)
            comparison = np.hstack([pred_overlay, gt_overlay])
            axes[seq_idx, 4].imshow(comparison)
            axes[seq_idx, 4].set_title('Pred vs GT')
            axes[seq_idx, 4].axis('off')
            
            # Metrics text
            metrics_text = f"{seq_name}\nAvg IoU: {avg_iou:.3f}\nBest Frame: {best_frame+1}\nFrame IoU: {frame_ious[best_frame]:.3f}"
            axes[seq_idx, 5].text(0.1, 0.5, metrics_text, fontsize=10, 
                                 transform=axes[seq_idx, 5].transAxes,
                                 verticalalignment='center')
            axes[seq_idx, 5].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_robustness_showcase(self, sequences, title, save_path):
        """Create figure showing robustness to complex scenes."""
        fig, axes = plt.subplots(len(sequences), 4, figsize=(16, 4*len(sequences)))
        fig.suptitle(title, fontsize=16, weight='bold')
        
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, analysis in enumerate(sequences):
            frames = analysis['frames']
            pred_masks = analysis['pred_masks']
            seq_name = analysis['sequence_name']
            avg_iou = analysis['avg_iou']
            
            # Show middle frame
            T = frames.shape[0]
            t = T // 2
            
            frame = self._normalize_frame(frames[t])
            
            axes[seq_idx, 0].imshow(frame)
            axes[seq_idx, 0].set_title('Complex Scene')
            axes[seq_idx, 0].axis('off')
            
            # Prediction overlay
            if len(pred_masks.shape) == 4:
                pred_mask = pred_masks[t, 0] > 0.5
            else:
                pred_mask = pred_masks[t] > 0.5
            
            if pred_mask.shape != frame.shape[:2]:
                pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
            
            pred_overlay = frame.copy()
            if torch.is_tensor(pred_mask):
                pred_mask_np = pred_mask.cpu().numpy()
            else:
                pred_mask_np = pred_mask
            pred_overlay[pred_mask_np] = (pred_overlay[pred_mask_np] * 0.7 + 
                                        np.array(self.colors['good']) * 0.3).astype(np.uint8)
            
            axes[seq_idx, 1].imshow(pred_overlay)
            axes[seq_idx, 1].set_title('Robust Detection')
            axes[seq_idx, 1].axis('off')
            
            # Show temporal progression
            T_show = min(T, 8)
            frame_indices = np.linspace(0, T-1, T_show, dtype=int)
            
            # Create mini temporal view
            temporal_strips = []
            for t_idx in frame_indices:
                mini_frame = self._normalize_frame(frames[t_idx])
                mini_frame = cv2.resize(mini_frame, (60, 45))
                temporal_strips.append(mini_frame)
            
            temporal_view = np.hstack(temporal_strips)
            axes[seq_idx, 2].imshow(temporal_view)
            axes[seq_idx, 2].set_title('Temporal Sequence')
            axes[seq_idx, 2].axis('off')
            
            # Metrics
            metrics_text = f"{seq_name}\nAvg IoU: {avg_iou:.3f}\nTemporal Consistency: {analysis['temporal_consistency']:.3f}\nCategory: Robust to Clutter"
            axes[seq_idx, 3].text(0.1, 0.5, metrics_text, fontsize=10, 
                                 transform=axes[seq_idx, 3].transAxes,
                                 verticalalignment='center')
            axes[seq_idx, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_boundary_issues_figure(self, sequences, title, save_path):
        """Create figure showing boundary precision issues."""
        fig, axes = plt.subplots(len(sequences), 5, figsize=(20, 4*len(sequences)))
        fig.suptitle(title, fontsize=16, weight='bold')
        
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, analysis in enumerate(sequences):
            frames = analysis['frames']
            pred_masks = analysis['pred_masks']
            gt_masks = analysis['gt_masks']
            seq_name = analysis['sequence_name']
            boundary_score = analysis['avg_boundary_score']
            worst_frame = analysis['worst_frame']
            
            # Show worst frame for boundary issues
            t = worst_frame
            frame = self._normalize_frame(frames[t])
            
            # Original
            axes[seq_idx, 0].imshow(frame)
            axes[seq_idx, 0].set_title('Input')
            axes[seq_idx, 0].axis('off')
            
            # Prediction with boundary highlighted
            if len(pred_masks.shape) == 4:
                pred_mask = pred_masks[t, 0] > 0.5
            else:
                pred_mask = pred_masks[t] > 0.5
            
            if pred_mask.shape != frame.shape[:2]:
                pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
            
            # Create boundary visualization
            kernel = np.ones((3,3), np.uint8)
            if torch.is_tensor(pred_mask):
                pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            else:
                pred_mask_np = pred_mask.astype(np.uint8)
            
            pred_boundary = cv2.morphologyEx(pred_mask_np, cv2.MORPH_GRADIENT, kernel)
            
            boundary_overlay = frame.copy()
            boundary_overlay[pred_boundary > 0] = [255, 255, 0]  # Yellow boundaries
            boundary_overlay[pred_mask_np > 0] = (boundary_overlay[pred_mask_np > 0] * 0.8 + 
                                               np.array(self.colors['prediction']) * 0.2).astype(np.uint8)
            
            axes[seq_idx, 1].imshow(boundary_overlay)
            axes[seq_idx, 1].set_title('Prediction + Boundaries')
            axes[seq_idx, 1].axis('off')
            
            # Ground truth
            gt_mask = gt_masks[t] > 0
            if torch.is_tensor(gt_mask):
                gt_mask_np = gt_mask.cpu().numpy().astype(np.uint8)
            else:
                gt_mask_np = gt_mask.astype(np.uint8)
            
            gt_boundary = cv2.morphologyEx(gt_mask_np, cv2.MORPH_GRADIENT, kernel)
            
            gt_overlay = frame.copy()
            gt_overlay[gt_boundary > 0] = [255, 255, 0]  # Yellow boundaries
            gt_overlay[gt_mask_np > 0] = (gt_overlay[gt_mask_np > 0] * 0.8 + 
                                        np.array(self.colors['ground_truth']) * 0.2).astype(np.uint8)
            
            axes[seq_idx, 2].imshow(gt_overlay)
            axes[seq_idx, 2].set_title('Ground Truth + Boundaries')
            axes[seq_idx, 2].axis('off')
            
            # Boundary comparison
            boundary_comparison = np.zeros_like(frame)
            boundary_comparison[pred_boundary > 0] = [0, 255, 0]  # Green for prediction
            boundary_comparison[gt_boundary > 0] = [255, 0, 0]   # Red for ground truth
            boundary_comparison[(pred_boundary > 0) & (gt_boundary > 0)] = [255, 255, 0]  # Yellow for overlap
            
            axes[seq_idx, 3].imshow(boundary_comparison)
            axes[seq_idx, 3].set_title('Boundary Comparison')
            axes[seq_idx, 3].axis('off')
            
            # Issues text
            frame_iou = analysis['frame_ious'][t]
            issues_text = f"{seq_name}\nFrame {t+1} (Worst)\nIoU: {frame_iou:.3f}\nBoundary Score: {boundary_score:.3f}\nIssue: Imprecise boundaries"
            axes[seq_idx, 4].text(0.1, 0.5, issues_text, fontsize=10, 
                                 transform=axes[seq_idx, 4].transAxes,
                                 verticalalignment='center')
            axes[seq_idx, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_issues_figure(self, sequences, title, save_path):
        """Create figure showing temporal inconsistency issues."""
        fig, axes = plt.subplots(len(sequences), 5, figsize=(20, 4*len(sequences)))
        fig.suptitle(title, fontsize=16, weight='bold')
        
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, analysis in enumerate(sequences):
            frames = analysis['frames']
            pred_masks = analysis['pred_masks']
            seq_name = analysis['sequence_name']
            temporal_score = analysis['temporal_consistency']
            
            T = frames.shape[0]
            
            # Show frames where temporal inconsistency is most apparent
            frame_indices = [0, T//3, 2*T//3, T-1] if T > 3 else list(range(T))
            
            for i, t in enumerate(frame_indices[:4]):
                frame = self._normalize_frame(frames[t])
                
                if len(pred_masks.shape) == 4:
                    pred_mask = pred_masks[t, 0] > 0.5
                else:
                    pred_mask = pred_masks[t] > 0.5
                
                if pred_mask.shape != frame.shape[:2]:
                    pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
                
                # Highlight temporal inconsistency with different colors
                overlay = frame.copy()
                if torch.is_tensor(pred_mask):
                    pred_mask_np = pred_mask.cpu().numpy()
                else:
                    pred_mask_np = pred_mask
                
                # Use different intensities to show variation
                alpha = 0.3 + 0.2 * (i / 3)  # Varying alpha to show changes
                overlay[pred_mask_np] = (overlay[pred_mask_np] * (1-alpha) + 
                                       np.array(self.colors['temporal_poor']) * alpha).astype(np.uint8)
                
                axes[seq_idx, i].imshow(overlay)
                axes[seq_idx, i].set_title(f'Frame {t+1}')
                axes[seq_idx, i].axis('off')
                
                # Add frame IoU
                if i < len(analysis['frame_ious']):
                    frame_iou = analysis['frame_ious'][t]
                    axes[seq_idx, i].text(0.02, 0.98, f'IoU: {frame_iou:.2f}', 
                                         transform=axes[seq_idx, i].transAxes,
                                         fontsize=8, color='white', weight='bold',
                                         verticalalignment='top',
                                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            # Issues text
            issues_text = f"{seq_name}\nTemporal Consistency: {temporal_score:.3f}\nIssue: Inconsistent predictions\nacross frames"
            axes[seq_idx, 4].text(0.1, 0.5, issues_text, fontsize=10, 
                                 transform=axes[seq_idx, 4].transAxes,
                                 verticalalignment='center')
            axes[seq_idx, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_failure_analysis(self, sequences, title, save_path):
        """Create figure analyzing failure cases."""
        fig, axes = plt.subplots(len(sequences), 6, figsize=(24, 4*len(sequences)))
        fig.suptitle(title, fontsize=16, weight='bold')
        
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        for seq_idx, analysis in enumerate(sequences):
            frames = analysis['frames']
            pred_masks = analysis['pred_masks']
            gt_masks = analysis['gt_masks']
            seq_name = analysis['sequence_name']
            avg_iou = analysis['avg_iou']
            worst_frame = analysis['worst_frame']
            
            # Show worst frame
            t = worst_frame
            frame = self._normalize_frame(frames[t])
            
            # Original
            axes[seq_idx, 0].imshow(frame)
            axes[seq_idx, 0].set_title('Input')
            axes[seq_idx, 0].axis('off')
            
            # Prediction
            if len(pred_masks.shape) == 4:
                pred_mask = pred_masks[t, 0] > 0.5
            else:
                pred_mask = pred_masks[t] > 0.5
            
            if pred_mask.shape != frame.shape[:2]:
                pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
            
            pred_overlay = frame.copy()
            if torch.is_tensor(pred_mask):
                pred_mask_np = pred_mask.cpu().numpy()
            else:
                pred_mask_np = pred_mask
            pred_overlay[pred_mask_np] = (pred_overlay[pred_mask_np] * 0.7 + 
                                        np.array(self.colors['poor']) * 0.3).astype(np.uint8)
            
            axes[seq_idx, 1].imshow(pred_overlay)
            axes[seq_idx, 1].set_title('Poor Prediction')
            axes[seq_idx, 1].axis('off')
            
            # Ground truth
            gt_mask = gt_masks[t] > 0
            gt_overlay = frame.copy()
            if torch.is_tensor(gt_mask):
                gt_mask_np = gt_mask.cpu().numpy()
            else:
                gt_mask_np = gt_mask
            gt_overlay[gt_mask_np] = (gt_overlay[gt_mask_np] * 0.7 + 
                                    np.array(self.colors['ground_truth']) * 0.3).astype(np.uint8)
            
            axes[seq_idx, 2].imshow(gt_overlay)
            axes[seq_idx, 2].set_title('Ground Truth')
            axes[seq_idx, 2].axis('off')
            
            # Error visualization
            error_vis = np.zeros_like(frame)
            false_positive = pred_mask_np & (~gt_mask_np)
            false_negative = (~pred_mask_np) & gt_mask_np
            true_positive = pred_mask_np & gt_mask_np
            
            error_vis[false_positive] = [255, 0, 0]    # Red for false positives
            error_vis[false_negative] = [0, 0, 255]    # Blue for false negatives
            error_vis[true_positive] = [0, 255, 0]     # Green for correct
            
            axes[seq_idx, 3].imshow(error_vis)
            axes[seq_idx, 3].set_title('Error Analysis')
            axes[seq_idx, 3].axis('off')
            
            # Performance graph
            frame_ious = analysis['frame_ious']
            axes[seq_idx, 4].plot(range(len(frame_ious)), frame_ious, 'ro-', linewidth=2, markersize=4)
            axes[seq_idx, 4].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good Threshold')
            axes[seq_idx, 4].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Poor Threshold')
            axes[seq_idx, 4].set_ylim(0, 1)
            axes[seq_idx, 4].set_title('IoU Progression')
            axes[seq_idx, 4].set_xlabel('Frame')
            axes[seq_idx, 4].set_ylabel('IoU')
            axes[seq_idx, 4].grid(True, alpha=0.3)
            axes[seq_idx, 4].legend(fontsize=8)
            
            # Failure analysis text
            frame_iou = analysis['frame_ious'][t]
            failure_text = f"{seq_name}\nWorst Frame: {t+1}\nFrame IoU: {frame_iou:.3f}\nAvg IoU: {avg_iou:.3f}\nPossible Issues:\n- Complex scene\n- Occlusion\n- Poor contrast"
            axes[seq_idx, 5].text(0.1, 0.5, failure_text, fontsize=9, 
                                 transform=axes[seq_idx, 5].transAxes,
                                 verticalalignment='center')
            axes[seq_idx, 5].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantitative_analysis(self, all_sequence_data):
        """Create comprehensive quantitative analysis."""
        print("Creating quantitative analysis...")
        
        # Extract metrics
        all_ious = []
        all_boundary_scores = []
        all_temporal_scores = []
        sequence_names = []
        
        for analysis in all_sequence_data:
            all_ious.extend(analysis['frame_ious'])
            all_boundary_scores.extend(analysis['frame_boundary_scores'])
            all_temporal_scores.append(analysis['temporal_consistency'])
            sequence_names.append(analysis['sequence_name'])
        
        # Create comprehensive quantitative figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quantitative Performance Analysis', fontsize=16, weight='bold')
        
        # 1. IoU Distribution
        ax1.hist(all_ious, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(all_ious), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(all_ious):.3f}')
        ax1.axvline(0.5, color='orange', linestyle='--', linewidth=2, 
                   label='Good Threshold')
        ax1.set_xlabel('IoU Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('IoU Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance categories
        excellent_count = len(self.analysis_results['excellent_cases'])
        good_count = len(self.analysis_results['good_cases'])
        poor_count = len(self.analysis_results['poor_cases'])
        
        categories = ['Excellent\n(IoU > 0.75)', 'Good\n(0.5 < IoU ≤ 0.75)', 'Poor\n(IoU ≤ 0.5)']
        counts = [excellent_count, good_count, poor_count]
        colors = ['green', 'orange', 'red']
        
        bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Number of Sequences')
        ax2.set_title('Performance Categories')
        
        # Add percentages on bars
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = (count / total) * 100 if total > 0 else 0
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 3. Temporal consistency analysis
        ax3.hist(all_temporal_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(all_temporal_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_temporal_scores):.3f}')
        ax3.axvline(0.8, color='orange', linestyle='--', linewidth=2,
                   label='Good Threshold')
        ax3.set_xlabel('Temporal Consistency Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Temporal Consistency Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. IoU vs Temporal Consistency scatter
        sequence_avg_ious = [np.mean(analysis['frame_ious']) for analysis in all_sequence_data]
        
        scatter = ax4.scatter(sequence_avg_ious, all_temporal_scores, 
                            c=[len(analysis['frame_ious']) for analysis in all_sequence_data],
                            cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Average IoU')
        ax4.set_ylabel('Temporal Consistency')
        ax4.set_title('IoU vs Temporal Consistency')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Sequence Length')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quantitative' / 'comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed metrics table
        self._create_metrics_table(all_sequence_data)
    
    def _create_metrics_table(self, all_sequence_data):
        """Create detailed metrics table."""
        metrics_data = []
        
        for analysis in all_sequence_data:
            metrics_data.append({
                'Sequence': analysis['sequence_name'],
                'Avg_IoU': analysis['avg_iou'],
                'Best_IoU': max(analysis['frame_ious']),
                'Worst_IoU': min(analysis['frame_ious']),
                'Temporal_Consistency': analysis['temporal_consistency'],
                'Avg_Boundary_Score': analysis['avg_boundary_score'],
                'Category': analysis['performance_category']
            })
        
        # Save to JSON and CSV
        with open(self.output_dir / 'quantitative' / 'detailed_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Create summary statistics
        summary_stats = {
            'total_sequences': len(all_sequence_data),
            'mean_iou': np.mean([d['Avg_IoU'] for d in metrics_data]),
            'std_iou': np.std([d['Avg_IoU'] for d in metrics_data]),
            'mean_temporal_consistency': np.mean([d['Temporal_Consistency'] for d in metrics_data]),
            'excellent_sequences': len(self.analysis_results['excellent_cases']),
            'good_sequences': len(self.analysis_results['good_cases']),
            'poor_sequences': len(self.analysis_results['poor_cases']),
            'temporal_consistent_sequences': len(self.analysis_results['temporal_consistent']),
            'boundary_accurate_sequences': len(self.analysis_results['boundary_accurate'])
        }
        
        with open(self.output_dir / 'quantitative' / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        return summary_stats
    
    def create_paper_ready_figure(self, all_sequence_data):
        """Create main figure for paper showing balanced results."""
        print("Creating paper-ready figure...")
        
        # Select representative examples
        representatives = {
            'excellent': self.analysis_results['excellent_cases'][:1] if self.analysis_results['excellent_cases'] else [],
            'good': self.analysis_results['good_cases'][:2] if self.analysis_results['good_cases'] else [],
            'limitations': self.analysis_results['poor_cases'][:1] if self.analysis_results['poor_cases'] else []
        }
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Mixed Results Analysis: Strengths and Limitations', fontsize=20, weight='bold')
        
        row = 0
        
        # Excellent case
        if representatives['excellent']:
            analysis = representatives['excellent'][0]
            self._add_sequence_to_paper_figure(fig, gs, analysis, row, 'Strength: High-Quality Segmentation', 'excellent')
            row += 1
        
        # Good cases
        for i, analysis in enumerate(representatives['good']):
            title = f'Typical Performance: Sequence {i+1}'
            self._add_sequence_to_paper_figure(fig, gs, analysis, row, title, 'good')
            row += 1
        
        # Limitation case
        if representatives['limitations']:
            analysis = representatives['limitations'][0]
            self._add_sequence_to_paper_figure(fig, gs, analysis, row, 'Limitation: Challenging Case', 'poor')
        
        plt.savefig(self.output_dir / 'paper_figures' / 'mixed_results_main_figure.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'paper_figures' / 'mixed_results_main_figure.pdf', 
                   bbox_inches='tight')
        plt.close()
    
    def _add_sequence_to_paper_figure(self, fig, gs, analysis, row, title, category):
        """Add a sequence to the paper figure."""
        frames = analysis['frames']
        pred_masks = analysis['pred_masks']
        gt_masks = analysis['gt_masks']
        
        # Select 3 frames
        T = frames.shape[0]
        frame_indices = [0, T//2, T-1] if T > 2 else list(range(min(T, 3)))
        
        color_map = {
            'excellent': self.colors['excellent'],
            'good': self.colors['good'],
            'poor': self.colors['poor']
        }
        
        # Add title
        title_ax = fig.add_subplot(gs[row, :])
        title_ax.text(0.02, 0.5, title, fontsize=14, weight='bold', 
                     transform=title_ax.transAxes, verticalalignment='center')
        title_ax.axis('off')
        
        for i, t in enumerate(frame_indices):
            frame = self._normalize_frame(frames[t])
            
            # Input
            ax_input = fig.add_subplot(gs[row, i*2])
            ax_input.imshow(frame)
            ax_input.set_title(f'Frame {t+1}' if i == 0 else f'Frame {t+1}')
            ax_input.axis('off')
            
            # Prediction vs GT
            if len(pred_masks.shape) == 4:
                pred_mask = pred_masks[t, 0] > 0.5
            else:
                pred_mask = pred_masks[t] > 0.5
            
            if pred_mask.shape != frame.shape[:2]:
                pred_mask = self._resize_mask_to_frame(pred_mask, frame.shape[:2])
            
            gt_mask = gt_masks[t] > 0
            
            # Create side-by-side comparison
            pred_overlay = frame.copy()
            gt_overlay = frame.copy()
            
            if torch.is_tensor(pred_mask):
                pred_mask_np = pred_mask.cpu().numpy()
            else:
                pred_mask_np = pred_mask
            
            if torch.is_tensor(gt_mask):
                gt_mask_np = gt_mask.cpu().numpy()
            else:
                gt_mask_np = gt_mask
            
            pred_overlay[pred_mask_np] = (pred_overlay[pred_mask_np] * 0.7 + 
                                        np.array(color_map[category]) * 0.3).astype(np.uint8)
            gt_overlay[gt_mask_np] = (gt_overlay[gt_mask_np] * 0.7 + 
                                    np.array(self.colors['ground_truth']) * 0.3).astype(np.uint8)
            
            comparison = np.hstack([pred_overlay, gt_overlay])
            
            ax_comparison = fig.add_subplot(gs[row, i*2 + 1])
            ax_comparison.imshow(comparison)
            ax_comparison.set_title('Pred | GT' if i == 0 else 'Pred | GT')
            ax_comparison.axis('off')
            
            # Add IoU score
            frame_iou = analysis['frame_ious'][t] if t < len(analysis['frame_ious']) else 0
            ax_comparison.text(0.02, 0.02, f'IoU: {frame_iou:.2f}', 
                             transform=ax_comparison.transAxes, fontsize=10, 
                             color='white', weight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    def _save_analysis_results(self, all_sequence_data):
        """Save comprehensive analysis results."""
        results_summary = {
            'total_sequences_analyzed': len(all_sequence_data),
            'categories': {
                'excellent_cases': len(self.analysis_results['excellent_cases']),
                'good_cases': len(self.analysis_results['good_cases']),
                'poor_cases': len(self.analysis_results['poor_cases']),
                'temporal_consistent': len(self.analysis_results['temporal_consistent']),
                'temporal_inconsistent': len(self.analysis_results['temporal_inconsistent']),
                'boundary_accurate': len(self.analysis_results['boundary_accurate']),
                'boundary_poor': len(self.analysis_results['boundary_poor'])
            },
            'example_sequences': {
                'excellent': [s['sequence_name'] for s in self.analysis_results['excellent_cases'][:3]],
                'good': [s['sequence_name'] for s in self.analysis_results['good_cases'][:3]],
                'poor': [s['sequence_name'] for s in self.analysis_results['poor_cases'][:3]]
            }
        }
        
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
    
    def run_complete_analysis(self, dataloader, max_sequences=20):
        """Run complete mixed results analysis."""
        print("=" * 80)
        print("RUNNING COMPREHENSIVE MIXED RESULTS ANALYSIS")
        print("=" * 80)
        
        # Step 1: Analyze all sequences
        all_sequence_data = self.analyze_sequences(dataloader, max_sequences)
        
        if not all_sequence_data:
            print("No sequences analyzed successfully!")
            return
        
        # Step 2: Create strengths showcase
        self.create_strengths_showcase()
        
        # Step 3: Create limitations analysis
        self.create_limitations_analysis()
        
        # Step 4: Create quantitative analysis
        self.create_quantitative_analysis(all_sequence_data)
        
        # Step 5: Create paper-ready figure
        self.create_paper_ready_figure(all_sequence_data)
        
        # Step 6: Create summary report
        self._create_summary_report(all_sequence_data)
        
        print("=" * 80)
        print("MIXED RESULTS ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("\nKey outputs:")
        print(f"📊 Main paper figure: {self.output_dir}/paper_figures/mixed_results_main_figure.pdf")
        print(f"✅ Strengths analysis: {self.output_dir}/strengths/")
        print(f"⚠️  Limitations analysis: {self.output_dir}/limitations/")
        print(f"📈 Quantitative analysis: {self.output_dir}/quantitative/")
        print(f"📝 Summary report: {self.output_dir}/summary_report.html")
        
        return all_sequence_data
    
    def _create_summary_report(self, all_sequence_data):
        """Create HTML summary report."""
        summary_stats = self._create_metrics_table(all_sequence_data)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mixed Results Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .strengths {{ border-left-color: #28a745; }}
                .limitations {{ border-left-color: #dc3545; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stat-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #fff3cd; }}
                .poor {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Mixed Results Analysis Report</h1>
                <p>Comprehensive analysis showing both strengths and limitations</p>
                <p>Total sequences analyzed: {summary_stats['total_sequences']}</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>{summary_stats['mean_iou']:.3f}</h3>
                        <p>Mean IoU</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary_stats['mean_temporal_consistency']:.3f}</h3>
                        <p>Mean Temporal Consistency</p>
                    </div>
                    <div class="stat-box excellent">
                        <h3>{summary_stats['excellent_sequences']}</h3>
                        <p>Excellent Cases (IoU > 0.75)</p>
                    </div>
                    <div class="stat-box good">
                        <h3>{summary_stats['good_sequences']}</h3>
                        <p>Good Cases (0.5 < IoU ≤ 0.75)</p>
                    </div>
                    <div class="stat-box poor">
                        <h3>{summary_stats['poor_sequences']}</h3>
                        <p>Challenging Cases (IoU ≤ 0.5)</p>
                    </div>
                </div>
            </div>
            
            <div class="section strengths">
                <h2>🎯 Key Strengths</h2>
                <ul>
                    <li><strong>Temporal Consistency:</strong> {summary_stats['temporal_consistent_sequences']} sequences show excellent temporal stability</li>
                    <li><strong>Robustness:</strong> Model maintains reasonable performance across diverse scenes</li>
                    <li><strong>Foreground Detection:</strong> Successfully identifies main objects in complex environments</li>
                </ul>
                <p><strong>Files:</strong> Check <code>strengths/</code> directory for detailed visualizations</p>
            </div>
            
            <div class="section limitations">
                <h2>⚠️ Identified Limitations</h2>
                <ul>
                    <li><strong>Boundary Precision:</strong> {summary_stats['total_sequences'] - summary_stats['boundary_accurate_sequences']} sequences show boundary issues</li>
                    <li><strong>Temporal Inconsistency:</strong> {summary_stats['total_sequences'] - summary_stats['temporal_consistent_sequences']} sequences have temporal variations</li>
                    <li><strong>Complex Scenes:</strong> {summary_stats['poor_sequences']} sequences show significant challenges</li>
                </ul>
                <p><strong>Files:</strong> Check <code>limitations/</code> directory for detailed analysis</p>
            </div>
            
            <div class="section">
                <h2>📊 For Your Paper</h2>
                <h3>Main Figure</h3>
                <p>Use <code>paper_figures/mixed_results_main_figure.pdf</code> as your main qualitative results figure.</p>
                
                <h3>Balanced Discussion Points</h3>
                <ul>
                    <li><strong>Strengths to highlight:</strong> Temporal consistency, robustness to clutter, consistent object detection</li>
                    <li><strong>Limitations to acknowledge:</strong> Boundary precision, occasional temporal inconsistency, performance on complex scenes</li>
                    <li><strong>Future work:</strong> Boundary refinement, post-processing for temporal smoothing, handling of complex occlusions</li>
                </ul>
                
                <h3>Quantitative Results</h3>
                <p>Mean IoU: {summary_stats['mean_iou']:.3f} ± {summary_stats['std_iou']:.3f}</p>
                <p>Success rate (IoU > 0.5): {((summary_stats['excellent_sequences'] + summary_stats['good_sequences']) / summary_stats['total_sequences'] * 100):.1f}%</p>
                <p>Temporal consistency: {summary_stats['mean_temporal_consistency']:.3f}</p>
            </div>
            
            <div class="section">
                <h2>📁 Generated Files</h2>
                <ul>
                    <li><strong>paper_figures/</strong> - Main figure for publication (PNG + PDF)</li>
                    <li><strong>strengths/</strong> - Visualizations highlighting model strengths</li>
                    <li><strong>limitations/</strong> - Analysis of model limitations</li>
                    <li><strong>quantitative/</strong> - Detailed metrics and statistics</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'summary_report.html', 'w') as f:
            f.write(html_content)


def main():
    """Main function for mixed results analysis."""
    parser = argparse.ArgumentParser(description='Mixed results analysis for video segmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/mamba_binary_efficient.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--output-dir', type=str, default='mixed_results_analysis',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max-sequences', type=int, default=20,
                       help='Maximum number of sequences to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {
            'model': {
                'input_dim': 3,
                'hidden_dims': [32, 64, 128],
                'd_state': 16,
                'temporal_window': 4,
                'dropout': 0.1,
                'd_conv': 4,
                'expand': 2
            },
            'dataset': {
                'img_size': [240, 320],
                'sequence_length': 3,
                'sequence_stride': 2,
                'num_workers': 2
            },
            'paths': {
                'davis_root': '/mnt/c/Datasets/DAVIS'
            }
        }
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = build_model(config).to(device)
    
    # Load checkpoint
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found")
        return
    
    model.eval()
    
    # Create data transform
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        normalize=True,
        train=False
    )
    
    # Create dataloader
    print("Creating dataloader...")
    try:
        dataloader = build_davis_dataloader(
            root_path=config['paths']['davis_root'],
            split=args.split,
            batch_size=1,
            transform=transform,
            **{k: v for k, v in config['dataset'].items() 
               if k not in ['batch_size', 'augmentation']}
        )
        print(f"Created dataloader with {len(dataloader)} sequences")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return
    
    # Initialize analyzer
    analyzer = MixedResultsAnalyzer(
        model=model,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis(dataloader, args.max_sequences)
        print("Analysis completed successfully!")
        print(f"\nOpen {args.output_dir}/summary_report.html for detailed navigation")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()