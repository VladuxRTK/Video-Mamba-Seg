import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from typing import Dict, List, Optional, Tuple, Union
import os


import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

class BinarySegmentationVisualizer:
    """Visualizes binary segmentation predictions and ground truth masks."""
    
    def __init__(self, save_dir='visualizations'):
        """Initialize visualizer with directory for saving visualizations."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_predictions(self, frames, pred_masks, gt_masks=None, save_path=None):
        """
        Create visualization of binary segmentation results.
        
        Args:
            frames: Video frames [B, T, C, H, W]
            pred_masks: Predicted segmentation masks [B, T, 1, H, W]
            gt_masks: Ground truth masks [B, T, H, W]
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure with visualization
        """
        # Take first batch item for visualization
        frames = frames[0]        # [T, C, H, W]
        pred_masks = pred_masks[0]  # [T, 1, H, W]
        if gt_masks is not None:
            gt_masks = gt_masks[0]  # [T, H, W]
        
        # Get number of frames and create figure
        T = frames.shape[0]
        cols = 3 if gt_masks is not None else 2
        fig, axes = plt.subplots(T, cols, figsize=(4*cols, 4*T))
        
        # Handle single frame case
        if T == 1:
            axes = np.array([axes]).reshape(1, -1)
        
        # Process each frame
        for t in range(T):
            # Show original frame
            frame_np = frames[t].permute(1, 2, 0).cpu().numpy()
            # Normalize if needed
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            
            axes[t, 0].imshow(frame_np)
            axes[t, 0].set_title(f"Frame {t+1}")
            axes[t, 0].axis('off')
            
            # Show predicted mask
            pred = pred_masks[t, 0].cpu().numpy()
            pred_binary = pred > 0.5
            
            # Create visualization with green overlay
            pred_vis = frame_np.copy()
            pred_vis[pred_binary] = pred_vis[pred_binary] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            axes[t, 1].imshow(pred_vis)
            axes[t, 1].set_title("Prediction")
            axes[t, 1].axis('off')
            
            # Show ground truth if available
            if gt_masks is not None:
                gt = gt_masks[t].cpu().numpy()
                gt_binary = gt > 0
                
                # Create visualization with green overlay
                gt_vis = frame_np.copy()
                gt_vis[gt_binary] = gt_vis[gt_binary] * 0.7 + np.array([0, 255, 0]) * 0.3
                
                axes[t, 2].imshow(gt_vis)
                axes[t, 2].set_title("Ground Truth")
                axes[t, 2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            plt.close()
        
        return fig
class VideoSegmentationVisualizer:
    """
    Comprehensive visualization tools for video instance segmentation results.
    Provides static images, videos, and analysis dashboards.
    """
    def __init__(
        self, 
        save_dir: Union[str, Path] = "visualizations",
        num_colors: int = 20,
        overlay_alpha: float = 0.6,
        figsize_per_frame: Tuple[int, int] = (5, 5)
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Create distinct colors for instances
        self.colors = self._generate_distinct_colors(num_colors)
        self.overlay_alpha = overlay_alpha
        self.figsize_per_frame = figsize_per_frame
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate visually distinct colors for instance visualization."""
        # Use tab20 colormap for up to 20 colors, then extend with hsv
        base_colors = plt.cm.tab20(np.linspace(0, 1, min(20, n)))[:, :3]
        
        if n <= 20:
            return base_colors
        
        # Add more colors if needed
        additional_colors = plt.cm.hsv(np.linspace(0, 1, n - 20))[:, :3]
        return np.vstack([base_colors, additional_colors])
    
    def visualize_frame(
        self,
        frame: torch.Tensor,            # [C, H, W]
        pred_masks: torch.Tensor,       # [N, H, W] or [1, H, W]
        gt_mask: Optional[torch.Tensor] = None,  # [H, W]
        frame_idx: int = 0,
        show_legend: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization of a single frame with predictions and ground truth.
        
        Args:
            frame: RGB frame tensor [C, H, W]
            pred_masks: Predicted instance masks [N, H, W] or binary mask [1, H, W]
            gt_mask: Optional ground truth mask [H, W]
            frame_idx: Index of the frame in the sequence
            show_legend: Whether to show color legend
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        # Convert inputs to numpy
        frame_np = frame.cpu().permute(1, 2, 0).numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        
        # Determine layout based on whether ground truth is provided
        n_cols = 3 if gt_mask is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(self.figsize_per_frame[0] * n_cols, 
                                                   self.figsize_per_frame[1]))
        
        # Plot original frame
        axes[0].imshow(frame_np)
        axes[0].set_title(f"Frame {frame_idx}")
        axes[0].axis('off')
        
        # Create prediction visualization
        pred_vis = frame_np.copy()
        instance_ids = []
        
        if pred_masks.shape[0] > 1:  # Multiple instances
            for i in range(pred_masks.shape[0]):
                mask = pred_masks[i].cpu().numpy() > 0.5
                if mask.any():  # Only if the mask has any positive pixels
                    color = self.colors[i % len(self.colors)]
                    pred_vis[mask] = self.overlay_alpha * np.array(color * 255) + \
                                    (1 - self.overlay_alpha) * pred_vis[mask]
                    instance_ids.append(i + 1)  # Instance IDs start from 1
        else:  # Binary mask case
            mask = pred_masks[0].cpu().numpy() > 0.5
            if mask.any():
                color = self.colors[0]
                pred_vis[mask] = self.overlay_alpha * np.array(color * 255) + \
                                (1 - self.overlay_alpha) * pred_vis[mask]
                instance_ids.append(1)
        
        # Plot predictions
        axes[1].imshow(pred_vis)
        axes[1].set_title("Prediction")
        axes[1].axis('off')
        
        # Add legend for instances
        if show_legend and instance_ids:
            patches = [
                mpatches.Patch(color=self.colors[i-1], label=f"Instance {i}")
                for i in instance_ids
            ]
            axes[1].legend(handles=patches, loc='upper right', fontsize='small')
        
        # Plot ground truth if provided
        if gt_mask is not None:
            gt_np = gt_mask.cpu().numpy()
            gt_vis = frame_np.copy()
            
            # Get unique instance IDs from ground truth
            unique_ids = np.unique(gt_np)
            unique_ids = unique_ids[unique_ids > 0]  # Skip background
            
            # Visualize each instance
            for i, idx in enumerate(unique_ids):
                mask = gt_np == idx
                color = self.colors[i % len(self.colors)]
                gt_vis[mask] = self.overlay_alpha * np.array(color * 255) + \
                              (1 - self.overlay_alpha) * gt_vis[mask]
            
            axes[2].imshow(gt_vis)
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
            
            # Add legend for ground truth
            if show_legend and len(unique_ids) > 0:
                patches = [
                    mpatches.Patch(color=self.colors[i % len(self.colors)], 
                                  label=f"GT {idx}")
                    for i, idx in enumerate(unique_ids)
                ]
                axes[2].legend(handles=patches, loc='upper right', fontsize='small')
        
        # Add overall title if provided
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        return fig
    
    def visualize_sequence(
        self,
        frames: torch.Tensor,            # [T, C, H, W]
        pred_masks: torch.Tensor,        # [T, N, H, W] or [T, 1, H, W]
        gt_masks: Optional[torch.Tensor] = None,  # [T, H, W]
        sequence_name: str = "sequence",
        max_frames: int = 8
    ) -> List[plt.Figure]:
        """
        Visualize a sequence of frames with predictions and ground truth.
        
        Args:
            frames: Sequence of frames [T, C, H, W]
            pred_masks: Predicted masks [T, N, H, W] or [T, 1, H, W]
            gt_masks: Optional ground truth masks [T, H, W]
            sequence_name: Name of the sequence
            max_frames: Maximum number of frames to visualize
            
        Returns:
            List of matplotlib figures
        """
        # Limit number of frames to visualize
        T = min(frames.shape[0], max_frames)
        figures = []
        
        # Create visualization for each frame
        for t in range(T):
            frame = frames[t]
            pred = pred_masks[t]
            gt = gt_masks[t] if gt_masks is not None else None
            
            fig = self.visualize_frame(
                frame=frame,
                pred_masks=pred,
                gt_mask=gt,
                frame_idx=t,
                title=f"{sequence_name} - Frame {t}"
            )
            figures.append(fig)
            
            # Save figure
            save_path = self.save_dir / f"{sequence_name}_frame_{t:03d}.png"
            fig.savefig(save_path)
            plt.close(fig)
        
        return figures
    
    def create_video(
        self,
        frames: torch.Tensor,            # [T, C, H, W]
        pred_masks: torch.Tensor,        # [T, N, H, W] or [T, 1, H, W]
        gt_masks: Optional[torch.Tensor] = None,  # [T, H, W]
        sequence_name: str = "sequence",
        fps: int = 10
    ) -> str:
        """
        Create a video visualization of the sequence.
        
        Args:
            frames: Sequence of frames [T, C, H, W]
            pred_masks: Predicted masks [T, N, H, W] or [T, 1, H, W]
            gt_masks: Optional ground truth masks [T, H, W]
            sequence_name: Name of the sequence
            fps: Frames per second for the video
            
        Returns:
            Path to the saved video file
        """
        T = frames.shape[0]
        
        # Create directory for frame images
        temp_dir = self.save_dir / f"{sequence_name}_frames"
        temp_dir.mkdir(exist_ok=True)
        
        # Create visualization for each frame and save as image
        for t in range(T):
            frame = frames[t]
            pred = pred_masks[t]
            gt = gt_masks[t] if gt_masks is not None else None
            
            fig = self.visualize_frame(
                frame=frame,
                pred_masks=pred,
                gt_mask=gt,
                frame_idx=t
            )
            
            # Save figure
            frame_path = temp_dir / f"frame_{t:03d}.png"
            fig.savefig(frame_path)
            plt.close(fig)
        
        # Create video from frames
        video_path = self.save_dir / f"{sequence_name}.mp4"
        
        # Get first image to determine size
        first_img = cv2.imread(str(temp_dir / "frame_000.png"))
        h, w, _ = first_img.shape
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        
        # Add frames to video
        for t in range(T):
            frame_path = temp_dir / f"frame_{t:03d}.png"
            img = cv2.imread(str(frame_path))
            video.write(img)
        
        # Release video writer
        video.release()
        
        # Option to clean up temporary frame images
        # for frame_path in temp_dir.glob("*.png"):
        #     frame_path.unlink()
        # temp_dir.rmdir()
        
        return str(video_path)
    
    def create_analysis_dashboard(
        self,
        frames: torch.Tensor,            # [T, C, H, W]
        pred_masks: torch.Tensor,        # [T, N, H, W]
        gt_masks: Optional[torch.Tensor] = None,  # [T, H, W]
        metrics: Optional[Dict[str, float]] = None,
        sequence_name: str = "sequence",
        save: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive analysis dashboard with visualizations and metrics.
        
        Args:
            frames: Sequence of frames [T, C, H, W]
            pred_masks: Predicted masks [T, N, H, W]
            gt_masks: Optional ground truth masks [T, H, W]
            metrics: Optional dictionary of metrics
            sequence_name: Name of the sequence
            save: Whether to save the dashboard
            
        Returns:
            Matplotlib figure with dashboard
        """
        T = frames.shape[0]
        num_instances = pred_masks.shape[1]
        
        # Select frames to display (first, middle, last)
        display_idxs = [0, T//2, T-1]
        if T <= 3:
            display_idxs = list(range(T))
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid for layout
        gs = fig.add_gridspec(3, 4)
        
        # Add title
        fig.suptitle(f"Analysis Dashboard: {sequence_name}", fontsize=16)
        
        # Plot selected frames
        for i, t in enumerate(display_idxs):
            if t >= T:
                continue
                
            # Get data for this frame
            frame = frames[t]
            pred = pred_masks[t]
            gt = gt_masks[t] if gt_masks is not None else None
            
            # Convert to numpy
            frame_np = frame.cpu().permute(1, 2, 0).numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            
            # Create subplot for this frame
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(frame_np)
            ax.set_title(f"Frame {t}")
            ax.axis('off')
            
            # Create prediction visualization
            ax_pred = fig.add_subplot(gs[i, 1])
            pred_vis = frame_np.copy()
            
            for n in range(num_instances):
                mask = pred[n].cpu().numpy() > 0.5
                if mask.any():
                    color = self.colors[n % len(self.colors)]
                    pred_vis[mask] = self.overlay_alpha * np.array(color * 255) + \
                                    (1 - self.overlay_alpha) * pred_vis[mask]
            
            ax_pred.imshow(pred_vis)
            ax_pred.set_title(f"Prediction {t}")
            ax_pred.axis('off')
            
            # Add ground truth if available
            if gt is not None:
                ax_gt = fig.add_subplot(gs[i, 2])
                gt_np = gt.cpu().numpy()
                gt_vis = frame_np.copy()
                
                unique_ids = np.unique(gt_np)
                unique_ids = unique_ids[unique_ids > 0]  # Skip background
                
                for i_gt, idx in enumerate(unique_ids):
                    mask = gt_np == idx
                    color = self.colors[i_gt % len(self.colors)]
                    gt_vis[mask] = self.overlay_alpha * np.array(color * 255) + \
                                  (1 - self.overlay_alpha) * gt_vis[mask]
                
                ax_gt.imshow(gt_vis)
                ax_gt.set_title(f"Ground Truth {t}")
                ax_gt.axis('off')
        
        # Add metrics panel if provided
        if metrics:
            ax_metrics = fig.add_subplot(gs[:, 3])
            ax_metrics.axis('off')
            
            # Create table of metrics
            metrics_text = "Evaluation Metrics:\n\n"
            for name, value in metrics.items():
                metrics_text += f"{name}: {value:.4f}\n"
            
            ax_metrics.text(0.1, 0.5, metrics_text, fontsize=12, 
                          verticalalignment='center')
            
            # Add temporal consistency analysis if available
            if 'temporal_consistency' in metrics:
                ax_metrics.text(0.1, 0.2, f"Temporal Consistency: Good", 
                              fontsize=12, color='green')
            
            # Add instance stability if available
            if 'instance_stability' in metrics:
                ax_metrics.text(0.1, 0.1, f"Instance Stability: Good", 
                              fontsize=12, color='green')
        
        plt.tight_layout()
        
        # Save dashboard
        if save:
            save_path = self.save_dir / f"{sequence_name}_dashboard.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def visualize_instance_tracking(
    frames: torch.Tensor,            # [T, C, H, W]
    pred_masks: torch.Tensor,        # [T, N, H, W]
    sequence_name: str = "sequence",
    save_dir: Union[str, Path] = "visualizations",
    track_instances: List[int] = None  # List of instance IDs to track
):
    """
    Create a visualization that shows how specific instances are tracked over time.
    
    Args:
        frames: Sequence of frames [T, C, H, W]
        pred_masks: Predicted masks [T, N, H, W]
        sequence_name: Name of the sequence
        save_dir: Directory to save visualizations
        track_instances: List of instance IDs to track (defaults to first 3)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    T, C, H, W = frames.shape
    _, N, _, _ = pred_masks.shape
    
    # Default to first 3 instances if not specified
    if track_instances is None:
        track_instances = list(range(min(3, N)))
    
    # Use different color for each instance
    colors = plt.cm.tab10(np.linspace(0, 1, len(track_instances)))
    
    # Create figure
    fig, axes = plt.subplots(len(track_instances), T, 
                           figsize=(T*3, len(track_instances)*3))
    
    # If only one instance, make axes 2D
    if len(track_instances) == 1:
        axes = axes.reshape(1, -1)
    
    # For each tracked instance
    for i, inst_id in enumerate(track_instances):
        for t in range(T):
            # Get frame and mask for this instance and frame
            frame = frames[t].cpu().permute(1, 2, 0).numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            mask = pred_masks[t, inst_id].cpu().numpy() > 0.5
            
            # Create visualization
            vis = frame.copy()
            if mask.any():
                color = colors[i, :3]
                overlay = np.zeros_like(vis)
                overlay[mask] = np.array(color * 255)
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            
            # Display in the appropriate subplot
            axes[i, t].imshow(vis)
            axes[i, t].set_title(f"Frame {t}")
            axes[i, t].axis('off')
        
        # Add instance label on the left
        axes[i, 0].set_ylabel(f"Instance {inst_id}", rotation=90, 
                           size='large', labelpad=15)
    
    plt.tight_layout()
    fig.suptitle(f"Instance Tracking: {sequence_name}", y=1.02)
    
    # Save visualization
    save_path = save_dir / f"{sequence_name}_instance_tracking.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path