import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from pathlib import Path

class TemporalStabilityAnalyzer:
    """
    Enhanced temporal stability analysis following DAVIS benchmark protocols.
    Computes frame-to-frame consistency, temporal smoothness, and stability decay.
    """
    
    def __init__(self, save_dir: str = "temporal_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_temporal_stability(
        self,
        pred_masks: torch.Tensor,  # [T, H, W] or [T, 1, H, W]
        gt_masks: Optional[torch.Tensor] = None,  # [T, H, W]
        flow: Optional[torch.Tensor] = None  # [T-1, 2, H, W]
    ) -> Dict[str, float]:
        """
        Compute comprehensive temporal stability metrics following DAVIS protocol.
        
        Args:
            pred_masks: Predicted masks over time
            gt_masks: Optional ground truth masks
            flow: Optional optical flow for motion-compensated evaluation
            
        Returns:
            Dictionary with temporal stability metrics
        """
        # Ensure binary masks
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)  # [T, H, W]
        
        # Convert to binary
        binary_pred = (pred_masks > 0.5).float()
        T, H, W = binary_pred.shape
        
        if T <= 1:
            return {'T_mean': 1.0, 'T_recall': 1.0, 'T_decay': 0.0}
        
        # 1. Frame-to-frame stability (basic temporal consistency)
        frame_to_frame_changes = []
        for t in range(T - 1):
            curr_mask = binary_pred[t]
            next_mask = binary_pred[t + 1]
            
            # Compute change ratio (lower is better)
            changed_pixels = (curr_mask != next_mask).float().sum()
            total_pixels = float(H * W)
            change_ratio = changed_pixels / total_pixels
            
            # Stability is 1 - change_ratio
            stability = 1.0 - change_ratio
            frame_to_frame_changes.append(stability.item())
        
        # 2. Motion-compensated stability if flow is available
        if flow is not None:
            motion_compensated_stability = self._compute_motion_compensated_stability(
                binary_pred, flow
            )
        else:
            motion_compensated_stability = np.mean(frame_to_frame_changes)
        
        # 3. Temporal decay (stability degradation over time)
        # Compare early frames vs late frames
        early_frames = frame_to_frame_changes[:len(frame_to_frame_changes)//3]
        late_frames = frame_to_frame_changes[-len(frame_to_frame_changes)//3:]
        
        if early_frames and late_frames:
            temporal_decay = max(0, np.mean(early_frames) - np.mean(late_frames))
        else:
            temporal_decay = 0.0
        
        # 4. Stability recall (percentage of frame transitions with high stability)
        stability_threshold = 0.9
        stable_transitions = sum(1 for s in frame_to_frame_changes if s >= stability_threshold)
        stability_recall = stable_transitions / len(frame_to_frame_changes)
        
        # 5. Overall temporal stability score (DAVIS T-measure)
        t_mean = np.mean(frame_to_frame_changes)
        
        return {
            'T_mean': t_mean,
            'T_recall': stability_recall,
            'T_decay': temporal_decay,
            'motion_compensated_T': motion_compensated_stability,
            'frame_to_frame_stabilities': frame_to_frame_changes
        }
    
    def _compute_motion_compensated_stability(
        self,
        masks: torch.Tensor,  # [T, H, W]
        flow: torch.Tensor    # [T-1, 2, H, W]
    ) -> float:
        """
        Compute motion-compensated temporal stability using optical flow.
        """
        compensated_stabilities = []
        T = masks.shape[0]
        
        for t in range(T - 1):
            curr_mask = masks[t]
            next_mask = masks[t + 1]
            flow_t = flow[t]  # [2, H, W]
            
            # Warp current mask to next frame using flow
            warped_mask = self._warp_mask(curr_mask, flow_t)
            
            # Compute stability between warped and actual next mask
            changed_pixels = (warped_mask != next_mask).float().sum()
            total_pixels = float(curr_mask.numel())
            stability = 1.0 - (changed_pixels / total_pixels)
            
            compensated_stabilities.append(stability.item())
        
        return np.mean(compensated_stabilities)
    
    def _warp_mask(self, mask: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp mask using optical flow.
        """
        H, W = mask.shape
        device = mask.device
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Add flow to coordinates
        new_x = x_coords + flow[0]  # flow[0] is x-component
        new_y = y_coords + flow[1]  # flow[1] is y-component
        
        # Normalize coordinates for grid_sample
        new_x_norm = 2.0 * new_x / (W - 1) - 1.0
        new_y_norm = 2.0 * new_y / (H - 1) - 1.0
        
        # Create sampling grid
        grid = torch.stack([new_x_norm, new_y_norm], dim=-1)  # [H, W, 2]
        
        # Warp mask
        warped = F.grid_sample(
            mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            grid.unsqueeze(0),               # [1, H, W, 2]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return (warped.squeeze() > 0.5).float()
    
    def visualize_temporal_stability(
        self,
        pred_masks: torch.Tensor,  # [T, H, W]
        sequence_name: str = "sequence",
        gt_masks: Optional[torch.Tensor] = None
    ) -> plt.Figure:
        """
        Create comprehensive temporal stability visualization.
        """
        # Compute stability metrics
        stability_metrics = self.compute_temporal_stability(pred_masks, gt_masks)
        frame_stabilities = stability_metrics['frame_to_frame_stabilities']
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Temporal Stability Analysis: {sequence_name}", fontsize=16)
        
        # 1. Frame-to-frame stability plot
        axes[0, 0].plot(frame_stabilities, 'b-', linewidth=2, marker='o')
        axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='High Stability Threshold')
        axes[0, 0].set_xlabel('Frame Transition')
        axes[0, 0].set_ylabel('Stability Score')
        axes[0, 0].set_title('Frame-to-Frame Stability')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Cumulative stability
        cumulative_stability = np.cumsum(frame_stabilities) / np.arange(1, len(frame_stabilities) + 1)
        axes[0, 1].plot(cumulative_stability, 'g-', linewidth=2)
        axes[0, 1].axhline(y=stability_metrics['T_mean'], color='orange', 
                          linestyle='--', label=f"Mean T-score: {stability_metrics['T_mean']:.3f}")
        axes[0, 1].set_xlabel('Frame Transition')
        axes[0, 1].set_ylabel('Cumulative Stability')
        axes[0, 1].set_title('Cumulative Temporal Stability')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Stability distribution
        axes[1, 0].hist(frame_stabilities, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=stability_metrics['T_mean'], color='red', 
                          linestyle='--', label=f"Mean: {stability_metrics['T_mean']:.3f}")
        axes[1, 0].set_xlabel('Stability Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Stability Score Distribution')
        axes[1, 0].legend()
        
        # 4. Metrics summary
        axes[1, 1].axis('off')
        metrics_text = "Temporal Stability Metrics:\n\n"
        metrics_text += f"T-mean: {stability_metrics['T_mean']:.4f}\n"
        metrics_text += f"T-recall: {stability_metrics['T_recall']:.4f}\n"
        metrics_text += f"T-decay: {stability_metrics['T_decay']:.4f}\n"
        if 'motion_compensated_T' in stability_metrics:
            metrics_text += f"Motion-comp. T: {stability_metrics['motion_compensated_T']:.4f}\n"
        
        # Add interpretation
        if stability_metrics['T_mean'] > 0.95:
            interpretation = "Excellent temporal stability"
            color = 'green'
        elif stability_metrics['T_mean'] > 0.90:
            interpretation = "Good temporal stability"
            color = 'orange'
        else:
            interpretation = "Poor temporal stability"
            color = 'red'
        
        metrics_text += f"\nInterpretation: {interpretation}"
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center', color=color)
        
        plt.tight_layout()
        
        # Save visualization
        save_path = self.save_dir / f"{sequence_name}_temporal_stability.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_temporal_consistency_video(
        self,
        frames: torch.Tensor,      # [T, C, H, W]
        pred_masks: torch.Tensor,  # [T, H, W]
        sequence_name: str = "sequence",
        fps: int = 10
    ) -> str:
        """
        Create a video showing temporal consistency with change detection.
        """
        T = frames.shape[0]
        
        # Create temporary directory for frames
        temp_dir = self.save_dir / f"{sequence_name}_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Process each frame
        for t in range(T):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original frame
            frame_np = frames[t].permute(1, 2, 0).cpu().numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            
            axes[0].imshow(frame_np)
            axes[0].set_title(f"Frame {t}")
            axes[0].axis('off')
            
            # Prediction overlay
            pred_np = (pred_masks[t] > 0.5).cpu().numpy()
            pred_overlay = frame_np.copy()
            pred_overlay[pred_np] = pred_overlay[pred_np] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            axes[1].imshow(pred_overlay)
            axes[1].set_title(f"Prediction {t}")
            axes[1].axis('off')
            
            # Temporal change visualization
            if t > 0:
                prev_pred = (pred_masks[t-1] > 0.5).cpu().numpy()
                curr_pred = (pred_masks[t] > 0.5).cpu().numpy()
                
                # Show changes
                change_vis = frame_np.copy()
                
                # Pixels that appeared (green)
                appeared = curr_pred & (~prev_pred)
                change_vis[appeared] = [0, 255, 0]
                
                # Pixels that disappeared (red)
                disappeared = prev_pred & (~curr_pred)
                change_vis[disappeared] = [255, 0, 0]
                
                # Stable pixels (blue overlay)
                stable = curr_pred & prev_pred
                change_vis[stable] = change_vis[stable] * 0.8 + np.array([0, 0, 255]) * 0.2
                
                axes[2].imshow(change_vis)
                axes[2].set_title(f"Changes from {t-1} to {t}")
            else:
                axes[2].imshow(frame_np)
                axes[2].set_title("No previous frame")
            
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save frame
            frame_path = temp_dir / f"frame_{t:03d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        # Create video
        video_path = self.save_dir / f"{sequence_name}_temporal_consistency.mp4"
        
        # Use OpenCV to create video
        first_img = cv2.imread(str(temp_dir / "frame_000.png"))
        h, w, _ = first_img.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        
        for t in range(T):
            frame_path = temp_dir / f"frame_{t:03d}.png"
            img = cv2.imread(str(frame_path))
            video.write(img)
        
        video.release()
        
        # Clean up temporary files
        for frame_path in temp_dir.glob("*.png"):
            frame_path.unlink()
        temp_dir.rmdir()
        
        return str(video_path)


def analyze_temporal_stability_for_sequence(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    sequence_name: Optional[str] = None,
    save_dir: str = "temporal_analysis"
) -> Dict[str, float]:
    """
    Analyze temporal stability for a specific sequence from the dataloader.
    """
    analyzer = TemporalStabilityAnalyzer(save_dir)
    model.eval()
    
    # Get one batch (sequence)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            gt_masks = batch['masks'].to(device)  # [B, T, H, W]
            seq_name = sequence_name or batch.get('sequence', [f"seq_{batch_idx}"])[0]
            
            # Get model predictions
            outputs = model(frames)
            pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
            
            # Take first batch item
            frames_seq = frames[0]  # [T, C, H, W]
            pred_seq = pred_masks[0]  # [T, 1, H, W] or [T, H, W]
            gt_seq = gt_masks[0]  # [T, H, W]
            
            # Compute temporal stability
            stability_metrics = analyzer.compute_temporal_stability(pred_seq, gt_seq)
            
            # Create visualizations
            print(f"Creating temporal stability analysis for {seq_name}...")
            
            # Stability plot
            analyzer.visualize_temporal_stability(pred_seq.squeeze(), seq_name, gt_seq)
            
            # Temporal consistency video
            video_path = analyzer.create_temporal_consistency_video(
                frames_seq, pred_seq.squeeze(), seq_name
            )
            
            print(f"Temporal stability metrics for {seq_name}:")
            for key, value in stability_metrics.items():
                if key != 'frame_to_frame_stabilities':
                    print(f"  {key}: {value:.4f}")
            
            print(f"Visualizations saved to: {save_dir}")
            print(f"Video saved to: {video_path}")
            
            return stability_metrics
    
    return {}


# Example usage function
def evaluate_temporal_stability_on_davis(
    model_path: str,
    config_path: str,
    sequence_name: Optional[str] = None,
    save_dir: str = "temporal_stability_results"
):
    """
    Evaluate temporal stability on DAVIS dataset for your VideoMamba model.
    """
    import yaml
    from models.binary_mamba_segmentation import build_model
    from datasets.davis import build_davis_dataloader
    from datasets.transforms import VideoSequenceAugmentation
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataloader
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        train=False
    )
    
    val_loader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split='val',
        batch_size=1,
        transform=transform,
        specific_sequence=sequence_name,  # Analyze specific sequence if provided
        **{k: v for k, v in config['dataset'].items() if k not in ['batch_size']}
    )
    
    # Analyze temporal stability
    results = analyze_temporal_stability_for_sequence(
        model=model,
        dataloader=val_loader,
        device=device,
        sequence_name=sequence_name,
        save_dir=save_dir
    )
    
    return results
