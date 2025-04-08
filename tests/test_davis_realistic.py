import torch
import time
from pathlib import Path
import sys

# Add parent directory to Python path so we can import our modules
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from datasets.davis import build_davis_dataloader
from models.model import build_model

class DAVISArchitectureTest:
    """
    Tests our video instance segmentation model on the DAVIS dataset.
    This testing framework helps us understand how our model performs
    on real-world video sequences.
    """
    def __init__(self, davis_root: str, model_config: Dict):
        self.davis_root = Path(davis_root)
        if not self.davis_root.exists():
            raise ValueError(f"DAVIS dataset not found at {davis_root}")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nInitializing DAVIS test environment:")
        print(f"- Using device: {self.device}")
        print(f"- DAVIS dataset path: {self.davis_root}")
        
        # Build and initialize model
        print("\nInitializing model...")
        self.model = build_model(model_config).to(self.device)
        self.model.eval()
        
        # Create output directory for visualizations
        self.output_dir = Path("davis_test_results")
        self.output_dir.mkdir(exist_ok=True)
        print(f"- Saving results to: {self.output_dir}")
    
    def test_sequence(self, sequence_name: str) -> Dict:
        """
        Tests the model on a single DAVIS sequence and computes metrics.
        Also tests model's ability to handle different batch configurations.
        """
        print(f"\nTesting sequence: {sequence_name}")
        
        # Test different configurations
        configs = [
            # (batch_size, seq_length, stride, img_size)
            (1, 4, 2, (240, 320)),  # Default configuration
            (2, 6, 3, (240, 320)),  # Longer sequence
            (1, 4, 2, (480, 640)),  # Higher resolution
        ]
        
        all_metrics = {}
        for config_idx, (batch_size, seq_length, stride, img_size) in enumerate(configs):
            print(f"\nTesting configuration {config_idx + 1}:")
            print(f"- Batch size: {batch_size}")
            print(f"- Sequence length: {seq_length}")
            print(f"- Stride: {stride}")
            print(f"- Image size: {img_size}")
            
            # Create dataloader for this configuration
            dataloader = build_davis_dataloader(
                root_path=str(self.davis_root),
                split='val',
                specific_sequence=sequence_name,
                batch_size=batch_size,
                img_size=img_size,
                sequence_length=seq_length,
                sequence_stride=stride,
                num_workers=2
            )
            
            total_batches = len(dataloader)
            print(f"Total batches: {total_batches}")
            
            # Initialize metrics
            batch_metrics = {
                'J_scores': [],
                'F_scores': [],
                'T_scores': [],
                'processing_times': [],
                'memory_usage': []
            }
            
            for batch_idx, batch in enumerate(dataloader):
                print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
                
                try:
                    # Move data to device
                    frames = batch['frames'].to(self.device)
                    masks = batch.get('masks')
                    if masks is not None:
                        masks = masks.to(self.device)
                    
                    # Record initial memory
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        initial_memory = torch.cuda.memory_allocated()
                    
                    # Time the forward pass
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = self.model(frames)
                    processing_time = time.time() - start_time
                    
                    # Get predictions
                    pred_masks = outputs['pred_masks']
                    
                    # Calculate metrics if ground truth is available
                    if masks is not None:
                        # J measure (IoU)
                        j_score = self._compute_j_measure(pred_masks, masks)
                        batch_metrics['J_scores'].append(j_score)
                        
                        # F measure (boundary)
                        f_score = self._compute_f_measure(pred_masks, masks)
                        batch_metrics['F_scores'].append(f_score)
                        
                        # T measure (temporal stability)
                        t_score = self._compute_temporal_stability(pred_masks)
                        batch_metrics['T_scores'].append(t_score)
                    
                    # Performance metrics
                    fps = frames.size(1) / processing_time
                    batch_metrics['processing_times'].append(processing_time)
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated()
                        memory_used = peak_memory - initial_memory
                        batch_metrics['memory_usage'].append(memory_used / 1e9)  # Convert to GB
                    
                    print(f"- Processing speed: {fps:.2f} FPS")
                    print(f"- Memory used: {memory_used/1e9:.2f} GB")
                    print(f"- Input shape: {frames.shape}")
                    print(f"- Output shape: {pred_masks.shape}")
                    
                    # Visualize first item in batch
                    self.visualize_results(
                        frames[0],
                        pred_masks[0],
                        masks[0] if masks is not None else None,
                        f"{sequence_name}_config{config_idx}_batch{batch_idx}"
                    )
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Compute average metrics for this configuration
            config_metrics = {
                'J_mean': np.mean(batch_metrics['J_scores']) if batch_metrics['J_scores'] else None,
                'F_mean': np.mean(batch_metrics['F_scores']) if batch_metrics['F_scores'] else None,
                'T_mean': np.mean(batch_metrics['T_scores']) if batch_metrics['T_scores'] else None,
                'avg_fps': len(batch_metrics['processing_times']) / sum(batch_metrics['processing_times']),
                'avg_memory': np.mean(batch_metrics['memory_usage']) if batch_metrics['memory_usage'] else None,
                'input_shape': frames.shape,
                'output_shape': pred_masks.shape
            }
            
            all_metrics[f'config_{config_idx}'] = config_metrics
            print(f"\nConfiguration {config_idx} metrics:")
            for metric_name, value in config_metrics.items():
                if value is not None:
                    print(f"- {metric_name}: {value}")
        
        print(f"\nCompleted processing sequence {sequence_name}")
        return all_metrics

    def _compute_j_measure(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute J measure (IoU)."""
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        return (intersection.float() / (union + 1e-6)).item()

    def _compute_f_measure(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute F measure (boundary similarity)."""
        pred_boundary = self._get_boundary(pred)
        gt_boundary = self._get_boundary(gt)
        
        # Compute precision and recall
        precision = (pred_boundary & gt_boundary).sum() / (pred_boundary.sum() + 1e-6)
        recall = (pred_boundary & gt_boundary).sum() / (gt_boundary.sum() + 1e-6)
        
        # Compute F measure
        return (2 * precision * recall / (precision + recall + 1e-6)).item()

    def _compute_temporal_stability(self, pred_masks: torch.Tensor) -> float:
        """Compute temporal stability score."""
        if pred_masks.dim() < 4:
            return 0.0
            
        stability_scores = []
        for t in range(pred_masks.shape[0] - 1):
            stability = (pred_masks[t] == pred_masks[t + 1]).float().mean()
            stability_scores.append(stability.item())
        
        return np.mean(stability_scores)
    
    def visualize_results(
    self,
    frames: torch.Tensor,    # [T, C, H, W]
    pred_masks: torch.Tensor,  # [T, N, H, W]
    gt_masks: Optional[torch.Tensor] = None,  # [T, H, W]
    save_name: str = "results"
):
        """
        Creates visualizations of the model's predictions with proper batch handling.
        """
        plt.close('all')
        
        # Convert tensors to numpy with explicit handling of dimensions
        frames = frames.cpu().numpy().transpose(0, 2, 3, 1)  # [T, H, W, C]
        
        # Handle pred_masks - ensure we get the right number of instances
        pred_masks = pred_masks.cpu().numpy()  # [T, N, H, W]
        num_frames = frames.shape[0]
        
        # Handle ground truth if provided
        if gt_masks is not None:
            gt_masks = gt_masks.cpu().numpy()  # [T, H, W]
            assert gt_masks.shape[0] == num_frames, "Ground truth frames don't match"
        
        # Create figure
        rows = 3 if gt_masks is not None else 2
        fig, axes = plt.subplots(rows, num_frames, figsize=(4*num_frames, 4*rows))
        if num_frames == 1:
            axes = axes.reshape(rows, 1)
        
        try:
            for t in range(num_frames):
                # Show original frame
                axes[0, t].imshow(frames[t])
                axes[0, t].set_title(f'Frame {t}')
                axes[0, t].axis('off')
                
                # Show predicted instances
                instance_viz = np.zeros_like(frames[t])
                num_instances = pred_masks.shape[1]
                for i in range(num_instances):
                    # Ensure mask has correct shape
                    mask = pred_masks[t, i]  # [H, W]
                    assert mask.shape == frames[t].shape[:2], \
                        f"Mask shape {mask.shape} doesn't match frame shape {frames[t].shape[:2]}"
                    
                    # Apply threshold to get binary mask
                    mask_bool = mask > 0.5
                    if mask_bool.any():
                        color = plt.cm.rainbow(i / num_instances)[:3]
                        instance_viz[mask_bool] = color
                
                axes[1, t].imshow(instance_viz)
                axes[1, t].set_title(f'Predictions')
                axes[1, t].axis('off')
                
                # Show ground truth if available
                if gt_masks is not None:
                    gt_viz = np.zeros_like(frames[t])
                    unique_ids = np.unique(gt_masks[t])[1:]  # Skip background
                    for i, idx in enumerate(unique_ids):
                        mask = gt_masks[t] == idx
                        color = plt.cm.rainbow(i / len(unique_ids))[:3]
                        gt_viz[mask] = color
                    
                    axes[2, t].imshow(gt_viz)
                    axes[2, t].set_title('Ground Truth')
                    axes[2, t].axis('off')
            
            plt.tight_layout()
            # Ensure the output directory exists
            save_path = self.output_dir / f"{save_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            print(f"Shapes - Frames: {frames.shape}, Predictions: {pred_masks.shape}")
            if gt_masks is not None:
                print(f"GT Masks: {gt_masks.shape}")
        
        finally:
            plt.close(fig)

def test_on_davis():
    """
    Main function to test our model on the DAVIS dataset.
    """
    # Create a complete test configuration
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'mask2former': {
            'hidden_dim': 256,          # Dimension of hidden features
            'num_queries': 16,          # Number of instance queries
            'mask_dim': 256,           # Dimension of mask features
            'nheads': 8,               # Number of attention heads
            'dim_feedforward': 1024,    # Dimension of feedforward network
            'dec_layers': 6,           # Number of decoder layers
            'enforce_input_project': False  # Whether to enforce input projection
        }
    }
    
    # Initialize tester with complete configuration
    davis_path = "/mnt/c/Datasets/DAVIS"
    tester = DAVISArchitectureTest(davis_path, config)
    
    # Test on a specific sequence
    sequence_name = "breakdance"
    metrics = tester.test_sequence(sequence_name)

    # Print metrics for each configuration
    print("\nResults Summary:")
    print("=" * 50)
    for config_name, config_metrics in metrics.items():
        print(f"\nConfiguration: {config_name}")
        print("-" * 30)
        
        # Print DAVIS metrics if available
        if config_metrics['J_mean'] is not None:
            print(f"DAVIS Metrics:")
            print(f"- J mean (IoU): {config_metrics['J_mean']:.3f}")
            print(f"- F mean (Boundary): {config_metrics['F_mean']:.3f}")
            print(f"- T mean (Temporal): {config_metrics['T_mean']:.3f}")
        
        # Print performance metrics
        print(f"\nPerformance Metrics:")
        print(f"- Average FPS: {config_metrics['avg_fps']:.2f}")
        print(f"- Average Memory Usage: {config_metrics['avg_memory']:.2f} GB")
        
        # Print shapes
        print(f"\nShapes:")
        print(f"- Input shape: {config_metrics['input_shape']}")
        print(f"- Output shape: {config_metrics['output_shape']}")
        print("-" * 30)
if __name__ == "__main__":
    test_on_davis()