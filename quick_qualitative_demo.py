#!/usr/bin/env python3
"""
Fixed Quick Qualitative Results Demo

Handles common issues like shape mismatches and normalization problems.
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

# Import your project modules
from models.binary_mamba_segmentation import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation


class FixedQuickQualitativeDemo:
    """Fixed qualitative results generator that handles shape mismatches and normalization."""
    
    def __init__(self, model, device='cuda', output_dir='quick_demo'):
        self.model = model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Simple color scheme
        self.pred_color = [0, 255, 0]  # Green for predictions
        self.gt_color = [255, 0, 0]    # Red for ground truth
        
        print(f"Demo initialized, output directory: {self.output_dir}")
    
    def _normalize_frame(self, frame):
        """Properly normalize frame for display."""
        # Convert from [C, H, W] to [H, W, C]
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = frame.permute(1, 2, 0)
        
        frame = frame.numpy()
        
        # Handle different normalization cases
        if frame.min() < 0 or frame.max() > 1:
            # Denormalize if it's been normalized with ImageNet stats
            # Typical ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            # Denormalize
            frame = frame * std + mean
            
            # Clip to valid range
            frame = np.clip(frame, 0, 1)
        
        # Convert to 0-255 range
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame
    
    def _resize_mask_to_frame(self, mask, target_shape):
        """Resize mask to match frame shape."""
        if len(mask.shape) == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif len(mask.shape) == 3:  # [1, H, W] or [H, W, 1]
            if mask.shape[0] == 1:  # [1, H, W]
                mask = mask.unsqueeze(0)  # [1, 1, H, W]
            else:  # [H, W, 1]
                mask = mask.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
        
        # Resize using nearest neighbor to preserve binary values
        resized_mask = F.interpolate(
            mask.float(), 
            size=target_shape, 
            mode='nearest'
        )
        
        return resized_mask.squeeze().bool()
    
    def _apply_overlay(self, frame, mask, color, alpha=0.3):
        """Apply colored overlay to frame where mask is True."""
        overlay = frame.copy()
        
        # Ensure mask matches frame spatial dimensions
        frame_h, frame_w = frame.shape[:2]
        if mask.shape != (frame_h, frame_w):
            print(f"Resizing mask from {mask.shape} to {(frame_h, frame_w)}")
            mask = self._resize_mask_to_frame(mask, (frame_h, frame_w))
        
        # Convert mask to numpy if it's a tensor
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # Apply overlay
        if mask.any():  # Only apply if there are positive pixels
            overlay[mask] = (overlay[mask] * (1 - alpha) + 
                           np.array(color) * alpha).astype(np.uint8)
        
        return overlay
    
    @torch.no_grad()
    def create_side_by_side_comparison(self, dataloader, num_sequences=3):
        """Create simple side-by-side comparison images with proper error handling."""
        print(f"Creating side-by-side comparisons for {num_sequences} sequences...")
        
        comparison_paths = []
        
        for seq_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            if seq_idx >= num_sequences:
                break
                
            try:
                # Get data
                frames = batch['frames'].to(self.device)
                masks = batch['masks'].to(self.device)
                sequence = batch.get('sequence', [f"seq_{seq_idx}"])[0]
                
                print(f"\nProcessing sequence: {sequence}")
                print(f"Input frames shape: {frames.shape}")
                print(f"Input masks shape: {masks.shape}")
                
                # Forward pass
                outputs = self.model(frames)
                pred_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                
                print(f"Output pred_masks shape: {pred_masks.shape}")
                
                # Move to CPU for processing
                frames = frames[0].cpu()  # [T, C, H, W]
                pred_masks = pred_masks[0].cpu()  # [T, 1, H, W] or [T, H, W]
                gt_masks = masks[0].cpu()  # [T, H, W]
                
                print(f"CPU frames shape: {frames.shape}")
                print(f"CPU pred_masks shape: {pred_masks.shape}")
                print(f"CPU gt_masks shape: {gt_masks.shape}")
                
                # Create comparison for this sequence
                comparison_path = self._create_sequence_comparison(
                    frames, pred_masks, gt_masks, sequence
                )
                comparison_paths.append(comparison_path)
                
                print(f"Successfully created comparison for {sequence}")
                
                # Clean up
                del frames, masks, outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing sequence {seq_idx}: {e}")
                print(f"Skipping sequence and continuing...")
                continue
        
        print(f"Successfully created {len(comparison_paths)} comparisons")
        return comparison_paths
    
    def _create_sequence_comparison(self, frames, pred_masks, gt_masks, seq_name):
        """Create a single comparison image for a sequence with robust error handling."""
        T = frames.shape[0]
        
        # Select frames to display
        num_frames = min(4, T)  # Show up to 4 frames
        if T > 1:
            frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
        else:
            frame_indices = [0]
            num_frames = 1
        
        print(f"Creating comparison for {seq_name} with {num_frames} frames")
        
        # Create figure
        fig, axes = plt.subplots(3, num_frames, figsize=(4*num_frames, 12))
        fig.suptitle(f'Sequence: {seq_name}', fontsize=16, weight='bold')
        
        # Handle single frame case
        if num_frames == 1:
            axes = axes.reshape(3, 1)
        
        for i, t in enumerate(frame_indices):
            try:
                # Get and normalize frame
                frame = self._normalize_frame(frames[t])
                print(f"Frame {t} normalized shape: {frame.shape}, range: [{frame.min()}, {frame.max()}]")
                
                # Original frame
                axes[0, i].imshow(frame)
                axes[0, i].set_title(f'Frame {t+1}')
                axes[0, i].axis('off')
                
                # Get prediction mask
                if len(pred_masks.shape) == 4:  # [T, 1, H, W]
                    pred_mask = pred_masks[t, 0] > 0.5
                elif len(pred_masks.shape) == 3:  # [T, H, W]
                    pred_mask = pred_masks[t] > 0.5
                else:
                    print(f"Unexpected pred_mask shape: {pred_masks.shape}")
                    pred_mask = pred_masks[t] > 0.5
                
                print(f"Pred mask shape: {pred_mask.shape}, positive pixels: {pred_mask.sum()}")
                
                # Apply prediction overlay
                pred_vis = self._apply_overlay(frame, pred_mask, self.pred_color)
                
                axes[1, i].imshow(pred_vis)
                axes[1, i].set_title('Prediction')
                axes[1, i].axis('off')
                
                # Ground truth overlay
                gt_mask = gt_masks[t] > 0
                print(f"GT mask shape: {gt_mask.shape}, positive pixels: {gt_mask.sum()}")
                
                gt_vis = self._apply_overlay(frame, gt_mask, self.gt_color)
                
                axes[2, i].imshow(gt_vis)
                axes[2, i].set_title('Ground Truth')
                axes[2, i].axis('off')
                
            except Exception as e:
                print(f"Error processing frame {t}: {e}")
                # Fill with placeholder
                axes[0, i].text(0.5, 0.5, f'Error\nFrame {t}', 
                               ha='center', va='center', transform=axes[0, i].transAxes)
                axes[1, i].text(0.5, 0.5, f'Error\nFrame {t}', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[2, i].text(0.5, 0.5, f'Error\nFrame {t}', 
                               ha='center', va='center', transform=axes[2, i].transAxes)
                for ax_idx in range(3):
                    axes[ax_idx, i].axis('off')
        
        # Add row labels
        axes[0, 0].text(-0.1, 0.5, 'Input', transform=axes[0, 0].transAxes,
                       rotation=90, va='center', ha='center', fontsize=14, weight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Prediction', transform=axes[1, 0].transAxes,
                       rotation=90, va='center', ha='center', fontsize=14, weight='bold')
        axes[2, 0].text(-0.1, 0.5, 'Ground Truth', transform=axes[2, 0].transAxes,
                       rotation=90, va='center', ha='center', fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f'{seq_name}_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison to: {save_path}")
        return save_path
    
    @torch.no_grad()
    def create_debug_info(self, dataloader):
        """Create debug information about the model and data."""
        print("Creating debug information...")
        
        debug_info = {
            'model_info': {},
            'data_info': {},
            'output_info': {}
        }
        
        # Get one batch for analysis
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Just analyze first batch
                break
                
            frames = batch['frames'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            # Model forward pass
            outputs = self.model(frames)
            
            # Collect debug info
            debug_info['data_info'] = {
                'frames_shape': list(frames.shape),
                'frames_dtype': str(frames.dtype),
                'frames_range': [float(frames.min()), float(frames.max())],
                'masks_shape': list(masks.shape),
                'masks_dtype': str(masks.dtype),
                'masks_range': [float(masks.min()), float(masks.max())],
                'num_sequences': len(batch.get('sequence', ['unknown']))
            }
            
            debug_info['output_info'] = {}
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    debug_info['output_info'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'range': [float(value.min()), float(value.max())]
                    }
            
            break
        
        # Save debug info
        import json
        debug_path = self.output_dir / 'debug_info.json'
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
        
        print("Debug information:")
        print(json.dumps(debug_info, indent=2))
        print(f"Debug info saved to: {debug_path}")
        
        return debug_info
    
    def run_quick_demo(self, dataloader):
        """Run complete quick demo with error handling."""
        print("=" * 60)
        print("RUNNING FIXED QUICK QUALITATIVE DEMO")
        print("=" * 60)
        
        # Create debug info first
        debug_info = self.create_debug_info(dataloader)
        
        results = {}
        
        try:
            # Side-by-side comparisons
            comparison_paths = self.create_side_by_side_comparison(dataloader, num_sequences=3)
            results['comparisons'] = comparison_paths
            
            # Create simple summary
            self._create_simple_summary(results, debug_info)
            
            print("=" * 60)
            print("QUICK DEMO COMPLETE")
            print("=" * 60)
            print(f"Results saved to: {self.output_dir}")
            print("Files created:")
            for path in comparison_paths:
                print(f"  📊 {path}")
            print(f"  🔍 {self.output_dir}/debug_info.json")
            print(f"  📝 {self.output_dir}/summary.txt")
            
        except Exception as e:
            print(f"Error during demo: {e}")
            print("Check debug_info.json for more details about the data shapes")
            
        return results
    
    def _create_simple_summary(self, results, debug_info):
        """Create a simple text summary with debug info."""
        summary_path = self.output_dir / 'summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("QUICK QUALITATIVE DEMO SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Data Information:\n")
            f.write(f"- Frame shape: {debug_info['data_info']['frames_shape']}\n")
            f.write(f"- Frame range: {debug_info['data_info']['frames_range']}\n")
            f.write(f"- Mask shape: {debug_info['data_info']['masks_shape']}\n")
            f.write(f"- Mask range: {debug_info['data_info']['masks_range']}\n\n")
            
            f.write("Model Output Information:\n")
            for key, value in debug_info['output_info'].items():
                f.write(f"- {key}: shape={value['shape']}, range={value['range']}\n")
            f.write("\n")
            
            f.write(f"Generated Files:\n")
            if 'comparisons' in results:
                f.write(f"- Sequence comparisons: {len(results['comparisons'])} files\n")
                for path in results['comparisons']:
                    f.write(f"  * {path.name}\n")
            
            f.write(f"\nAll files saved to: {self.output_dir}\n")


def main():
    """Main function for fixed quick qualitative demo."""
    parser = argparse.ArgumentParser(description='Fixed quick qualitative results demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/mamba_binary_efficient.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--output-dir', type=str, default='fixed_quick_demo',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max-sequences', type=int, default=3,
                       help='Maximum number of sequences to process')
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
        print(f"Warning: Checkpoint {args.checkpoint} not found, using untrained model")
    
    model.eval()
    
    # Create data transform (no augmentation)
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
    
    # Initialize demo generator
    demo = FixedQuickQualitativeDemo(
        model=model,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run demo
    try:
        results = demo.run_quick_demo(dataloader)
        print("Demo completed successfully!")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()