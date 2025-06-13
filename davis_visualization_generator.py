#!/usr/bin/env python3
"""
Generate real visualizations using actual DAVIS dataset images and your trained VideoMamba model.
This script loads real DAVIS sequences and creates compelling paper figures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
from pathlib import Path
import os
import sys
from tqdm import tqdm
import json

# Add your project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.binary_mamba_segmentation import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation

class DAVISVisualizationGenerator:
    """Generate visualizations using real DAVIS data and VideoMamba predictions."""
    
    def __init__(self, config_path, checkpoint_path=None, save_dir="real_davis_figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Build and load model
        self.model = build_model(self.config)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint loaded - using random weights")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup transforms
        self.transform = VideoSequenceAugmentation(
            img_size=tuple(self.config['dataset']['img_size']),
            train=False,
            normalize=True
        )
        
        print(f"📁 Visualizations will be saved to: {self.save_dir}")
    
    def load_davis_sequence(self, sequence_name, split='val', max_frames=8):
        """Load a specific DAVIS sequence."""
        
        try:
            # Create dataloader for specific sequence
            dataloader = build_davis_dataloader(
                root_path=self.config['paths']['davis_root'],
                split=split,
                batch_size=1,
                sequence_length=max_frames,
                sequence_stride=1,
                transform=self.transform,
                specific_sequence=sequence_name
            )
            
            # Get the first (and only) batch
            batch = next(iter(dataloader))
            
            print(f"✅ Loaded DAVIS sequence '{sequence_name}':")
            print(f"   - Frames shape: {batch['frames'].shape}")
            if 'masks' in batch:
                print(f"   - Masks shape: {batch['masks'].shape}")
            
            return batch
            
        except Exception as e:
            print(f"❌ Error loading sequence '{sequence_name}': {e}")
            return None
    
    def get_model_predictions(self, frames):
        """Get predictions from VideoMamba model."""
        
        with torch.no_grad():
            frames = frames.to(self.device)
            outputs = self.model(frames)
            
            # Use adaptive masks if available, otherwise regular predictions
            pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
            
            return {
                'pred_masks': pred_masks.cpu(),
                'logits': outputs['logits'].cpu() if 'logits' in outputs else None,
                'raw_predictions': outputs['pred_masks'].cpu()
            }
    
    def create_sequence_comparison_figure(self, sequence_name, split='val'):
        """Create a figure comparing VideoMamba predictions with ground truth."""
        
        # Load sequence
        batch = self.load_davis_sequence(sequence_name, split)
        if batch is None:
            return None
        
        frames = batch['frames']  # [1, T, C, H, W]
        gt_masks = batch.get('masks')  # [1, T, H, W] if available
        
        # Get predictions
        predictions = self.get_model_predictions(frames)
        pred_masks = predictions['pred_masks'][0]  # [T, 1, H, W]
        
        # Create visualization
        T = frames.shape[1]
        fig, axes = plt.subplots(3 if gt_masks is not None else 2, min(T, 6), 
                                figsize=(min(T, 6) * 4, 12 if gt_masks is not None else 8))
        
        if T == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'VideoMamba Results: DAVIS "{sequence_name}" Sequence', 
                     fontsize=16, fontweight='bold')
        
        # Process each frame
        for t in range(min(T, 6)):
            # Original frame
            frame = frames[0, t].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            axes[0, t].imshow(frame)
            axes[0, t].set_title(f'Frame {t+1}')
            axes[0, t].axis('off')
            
            # VideoMamba prediction
            pred = pred_masks[t, 0].numpy()
            pred_binary = pred > 0.5
            
            # Create overlay
            pred_vis = frame.copy()
            if pred_binary.any():
                pred_vis[pred_binary] = pred_vis[pred_binary] * 0.6 + np.array([0, 255, 0]) * 0.4
            
            axes[1, t].imshow(pred_vis)
            axes[1, t].set_title('VideoMamba Prediction')
            axes[1, t].axis('off')
            
            # Add confidence score
            confidence = pred.max() if pred.size > 0 else 0
            axes[1, t].text(5, 20, f'Conf: {confidence:.3f}', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           fontsize=10, fontweight='bold')
            
            # Ground truth (if available)
            if gt_masks is not None:
                gt = gt_masks[0, t].numpy()
                gt_binary = gt > 0
                
                gt_vis = frame.copy()
                if gt_binary.any():
                    gt_vis[gt_binary] = gt_vis[gt_binary] * 0.6 + np.array([0, 0, 255]) * 0.4
                
                axes[2, t].imshow(gt_vis)
                axes[2, t].set_title('Ground Truth')
                axes[2, t].axis('off')
                
                # Calculate IoU
                if pred_binary.any() and gt_binary.any():
                    intersection = (pred_binary & gt_binary).sum()
                    union = (pred_binary | gt_binary).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    axes[2, t].text(5, 20, f'IoU: {iou:.3f}', 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                                   fontsize=10, fontweight='bold')
        
        # Add temporal consistency analysis
        if T > 1:
            # Calculate frame-to-frame consistency
            consistencies = []
            for t in range(T-1):
                if t+1 < pred_masks.shape[0]:
                    diff = torch.abs(pred_masks[t+1, 0] - pred_masks[t, 0]).mean()
                    consistency = 1 - diff.item()
                    consistencies.append(consistency)
            
            avg_consistency = np.mean(consistencies) if consistencies else 0
            
            # Add text box with metrics
            textstr = f'Temporal Consistency: {avg_consistency:.3f}\n'
            textstr += f'Sequence Length: {T} frames\n'
            textstr += f'Avg Confidence: {pred_masks.mean():.3f}'
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            fig.text(0.02, 0.02, textstr, fontsize=12, bbox=props)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / f"{sequence_name}_comparison.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved comparison for '{sequence_name}': {save_path}")
        return save_path
    
    def create_temporal_consistency_showcase(self, sequences=['blackswan', 'bmx-trees', 'camel']):
        """Create a showcase of temporal consistency across multiple sequences."""
        
        fig, axes = plt.subplots(len(sequences), 6, figsize=(18, len(sequences) * 3))
        if len(sequences) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('VideoMamba Temporal Consistency Across DAVIS Sequences', 
                     fontsize=16, fontweight='bold')
        
        all_consistencies = []
        
        for seq_idx, sequence_name in enumerate(sequences):
            print(f"Processing sequence: {sequence_name}")
            
            # Load sequence
            batch = self.load_davis_sequence(sequence_name, max_frames=6)
            if batch is None:
                continue
            
            frames = batch['frames']  # [1, T, C, H, W]
            
            # Get predictions
            predictions = self.get_model_predictions(frames)
            pred_masks = predictions['pred_masks'][0]  # [T, 1, H, W]
            
            T = frames.shape[1]
            
            # Calculate temporal consistency
            consistencies = []
            for t in range(min(T, 6)):
                if t > 0:
                    diff = torch.abs(pred_masks[t, 0] - pred_masks[t-1, 0]).mean()
                    consistency = 1 - diff.item()
                    consistencies.append(consistency)
                
                # Visualize frame
                frame = frames[0, t].permute(1, 2, 0).numpy()
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                pred = pred_masks[t, 0].numpy()
                pred_binary = pred > 0.5
                
                # Create overlay
                vis = frame.copy()
                if pred_binary.any():
                    vis[pred_binary] = vis[pred_binary] * 0.7 + np.array([0, 255, 0]) * 0.3
                
                axes[seq_idx, t].imshow(vis)
                axes[seq_idx, t].axis('off')
                
                if seq_idx == 0:  # Add frame numbers on top row
                    axes[seq_idx, t].set_title(f'Frame {t+1}')
                
                # Add consistency score
                if t > 0 and consistencies:
                    score = consistencies[-1]
                    color = 'green' if score > 0.9 else 'orange' if score > 0.8 else 'red'
                    axes[seq_idx, t].text(5, 15, f'{score:.3f}', 
                                         bbox=dict(boxstyle="round,pad=0.2", 
                                                  facecolor=color, alpha=0.8),
                                         fontsize=9, fontweight='bold', color='white')
            
            # Add sequence label
            axes[seq_idx, 0].text(-0.1, 0.5, sequence_name, rotation=90, 
                                 transform=axes[seq_idx, 0].transAxes,
                                 ha='center', va='center', fontweight='bold', fontsize=12)
            
            # Store consistency scores
            if consistencies:
                all_consistencies.extend(consistencies)
        
        # Add overall consistency score
        if all_consistencies:
            avg_consistency = np.mean(all_consistencies)
            fig.text(0.5, 0.02, f'Overall Temporal Consistency: {avg_consistency:.3f}', 
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / "temporal_consistency_showcase.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved temporal consistency showcase: {save_path}")
        return save_path
    
    def create_failure_analysis_with_real_data(self, sequences=['tennis', 'car-shadow']):
        """Create honest failure analysis using real DAVIS data."""
        
        fig, axes = plt.subplots(2, len(sequences) * 3, figsize=(len(sequences) * 9, 6))
        fig.suptitle('VideoMamba: Real Performance Analysis on DAVIS', 
                     fontsize=16, fontweight='bold')
        
        for seq_idx, sequence_name in enumerate(sequences):
            print(f"Analyzing sequence: {sequence_name}")
            
            # Load sequence
            batch = self.load_davis_sequence(sequence_name, max_frames=3)
            if batch is None:
                continue
            
            frames = batch['frames']  # [1, T, C, H, W]
            gt_masks = batch.get('masks')  # [1, T, H, W]
            
            # Get predictions
            predictions = self.get_model_predictions(frames)
            pred_masks = predictions['pred_masks'][0]  # [T, 1, H, W]
            
            metrics = []
            
            for t in range(3):
                col_idx = seq_idx * 3 + t
                
                # Original frame
                frame = frames[0, t].permute(1, 2, 0).numpy()
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                # Prediction
                pred = pred_masks[t, 0].numpy()
                pred_binary = pred > 0.5
                
                # Ground truth
                if gt_masks is not None:
                    gt = gt_masks[0, t].numpy()
                    gt_binary = gt > 0
                    
                    # Calculate metrics
                    if pred_binary.any() and gt_binary.any():
                        intersection = (pred_binary & gt_binary).sum()
                        union = (pred_binary | gt_binary).sum()
                        iou = intersection / union if union > 0 else 0
                        
                        # Boundary F-score (simplified)
                        pred_boundary = cv2.Canny((pred * 255).astype(np.uint8), 50, 150)
                        gt_boundary = cv2.Canny((gt * 255).astype(np.uint8), 50, 150)
                        boundary_intersection = (pred_boundary > 0) & (gt_boundary > 0)
                        boundary_union = (pred_boundary > 0) | (gt_boundary > 0)
                        f_score = (2 * boundary_intersection.sum()) / (boundary_union.sum() + pred_boundary.sum() + 1e-6)
                        
                        metrics.append({'iou': iou, 'f_score': f_score})
                    else:
                        metrics.append({'iou': 0, 'f_score': 0})
                
                # Visualize prediction
                pred_vis = frame.copy()
                if pred_binary.any():
                    pred_vis[pred_binary] = pred_vis[pred_binary] * 0.7 + np.array([0, 255, 0]) * 0.3
                
                axes[0, col_idx].imshow(pred_vis)
                axes[0, col_idx].set_title(f'{sequence_name} F{t+1}')
                axes[0, col_idx].axis('off')
                
                # Show ground truth if available
                if gt_masks is not None:
                    gt_vis = frame.copy()
                    if gt_binary.any():
                        gt_vis[gt_binary] = gt_vis[gt_binary] * 0.7 + np.array([0, 0, 255]) * 0.3
                    
                    axes[1, col_idx].imshow(gt_vis)
                    axes[1, col_idx].set_title('Ground Truth')
                    axes[1, col_idx].axis('off')
                    
                    # Add metrics
                    if metrics and len(metrics) > t:
                        metric_text = f"IoU: {metrics[t]['iou']:.3f}\nF: {metrics[t]['f_score']:.3f}"
                        axes[1, col_idx].text(5, 20, metric_text,
                                            bbox=dict(boxstyle="round,pad=0.3", 
                                                     facecolor='white', alpha=0.8),
                                            fontsize=10, fontweight='bold')
        
        # Add row labels
        axes[0, 0].text(-0.15, 0.5, 'VideoMamba\nPredictions', rotation=90,
                       transform=axes[0, 0].transAxes, ha='center', va='center',
                       fontweight='bold', fontsize=12)
        
        if gt_masks is not None:
            axes[1, 0].text(-0.15, 0.5, 'Ground\nTruth', rotation=90,
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           fontweight='bold', fontsize=12)
        
        # Add performance summary
        if metrics:
            avg_iou = np.mean([m['iou'] for m in metrics])
            avg_f = np.mean([m['f_score'] for m in metrics])
            
            summary_text = f'Performance Summary:\n'
            summary_text += f'Average IoU: {avg_iou:.3f}\n'
            summary_text += f'Average F-score: {avg_f:.3f}\n'
            summary_text += f'Sequences: {", ".join(sequences)}'
            
            fig.text(0.02, 0.02, summary_text, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / "real_performance_analysis.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved real performance analysis: {save_path}")
        return save_path
    
    def create_video_demonstration(self, sequence_name='blackswan', output_fps=5):
        """Create a video showing VideoMamba predictions over time."""
        
        # Load sequence with more frames
        batch = self.load_davis_sequence(sequence_name, max_frames=20)
        if batch is None:
            return None
        
        frames = batch['frames']  # [1, T, C, H, W]
        gt_masks = batch.get('masks')  # [1, T, H, W]
        
        # Get predictions
        predictions = self.get_model_predictions(frames)
        pred_masks = predictions['pred_masks'][0]  # [T, 1, H, W]
        
        T = frames.shape[1]
        video_frames = []
        
        for t in range(T):
            # Create side-by-side visualization
            frame = frames[0, t].permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            pred = pred_masks[t, 0].numpy()
            pred_binary = pred > 0.5
            
            h, w = frame.shape[:2]
            
            if gt_masks is not None:
                # Three panels: original, prediction, ground truth
                combined = np.zeros((h, w*3, 3), dtype=np.uint8)
                
                # Original
                combined[:, :w] = frame
                
                # Prediction
                pred_vis = frame.copy()
                if pred_binary.any():
                    pred_vis[pred_binary] = pred_vis[pred_binary] * 0.7 + np.array([0, 255, 0]) * 0.3
                combined[:, w:2*w] = pred_vis
                
                # Ground truth
                gt = gt_masks[0, t].numpy()
                gt_binary = gt > 0
                gt_vis = frame.copy()
                if gt_binary.any():
                    gt_vis[gt_binary] = gt_vis[gt_binary] * 0.7 + np.array([0, 0, 255]) * 0.3
                combined[:, 2*w:] = gt_vis
                
                # Add labels
                cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, 'VideoMamba', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, 'Ground Truth', (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Two panels: original and prediction
                combined = np.zeros((h, w*2, 3), dtype=np.uint8)
                
                # Original
                combined[:, :w] = frame
                
                # Prediction
                pred_vis = frame.copy()
                if pred_binary.any():
                    pred_vis[pred_binary] = pred_vis[pred_binary] * 0.7 + np.array([0, 255, 0]) * 0.3
                combined[:, w:] = pred_vis
                
                # Add labels
                cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, 'VideoMamba', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add frame number
            cv2.putText(combined, f'Frame {t+1}/{T}', (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Add temporal consistency score
            if t > 0:
                diff = torch.abs(pred_masks[t, 0] - pred_masks[t-1, 0]).mean()
                consistency = 1 - diff.item()
                cv2.putText(combined, f'Consistency: {consistency:.3f}', 
                           (combined.shape[1] - 200, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            video_frames.append(combined)
        
        # Save video
        if video_frames:
            video_path = self.save_dir / f"{sequence_name}_demo.mp4"
            h, w = video_frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, output_fps, (w, h))
            
            for frame in video_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            print(f"✅ Saved video demonstration: {video_path}")
            
            return video_path
        
        return None
    
    def generate_all_real_visualizations(self):
        """Generate all visualizations using real DAVIS data."""
        
        print("🎬 Generating real DAVIS visualizations...")
        
        generated_files = []
        
        # 1. Individual sequence comparisons
        sequences = ['blackswan', 'bmx-trees', 'camel', 'car-shadow', 'cows']
        
        for seq in sequences[:3]:  # Limit to first 3 for demo
            print(f"\n📊 Creating comparison for '{seq}'...")
            path = self.create_sequence_comparison_figure(seq)
            if path:
                generated_files.append(path)
        
        # 2. Temporal consistency showcase
        print(f"\n⏱️ Creating temporal consistency showcase...")
        path = self.create_temporal_consistency_showcase(['blackswan', 'bmx-trees', 'camel'])
        if path:
            generated_files.append(path)
        
        # 3. Real performance analysis
        print(f"\n📈 Creating real performance analysis...")
        path = self.create_failure_analysis_with_real_data(['tennis', 'car-shadow'])
        if path:
            generated_files.append(path)
        
        # 4. Video demonstration
        print(f"\n🎥 Creating video demonstration...")
        path = self.create_video_demonstration('blackswan')
        if path:
            generated_files.append(path)
        
        return generated_files

def main():
    """Main function to generate real DAVIS visualizations."""
    
    print("🎯 VideoMamba Real DAVIS Visualization Generator")
    print("=" * 60)
    
    # Configuration
    config_path = "configs/mamba_binary_efficient.yaml"
    checkpoint_path = "checkpoints/mamba_binary/model_best.pth"  # Adjust as needed
    
    try:
        # Create generator
        generator = DAVISVisualizationGenerator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            save_dir="real_davis_figures"
        )
        
        # Generate all visualizations
        files = generator.generate_all_real_visualizations()
        
        print("\n" + "=" * 60)
        print("✅ All real DAVIS visualizations generated!")
        print(f"\n📁 Generated {len(files)} files:")
        
        for i, file_path in enumerate(files, 1):
            print(f"   {i:2d}. {file_path}")
        
        print(f"\n📂 All files saved to: {generator.save_dir}")
        
        print("\n💡 Usage tips:")
        print("   • Use sequence comparisons for individual results")
        print("   • Temporal consistency showcase for main figure")
        print("   • Performance analysis for honest limitations")
        print("   • Video demonstration for presentations")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   • Check if DAVIS dataset path is correct in config")
        print("   • Verify model checkpoint exists")
        print("   • Ensure DAVIS data is properly formatted")
        raise

if __name__ == "__main__":
    main()