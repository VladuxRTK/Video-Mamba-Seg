

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation


def visualize_sequence(frames, masks, sequence_name, save_dir=None):
    """Visualize a sequence of frames and masks."""
    # Convert frames from tensor [T, C, H, W] to numpy [T, H, W, C]
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    
    # Denormalize if needed
    if frames.max() <= 1.0:
        frames = frames * 255
    frames = frames.astype(np.uint8)
    
    if masks is not None:
        masks = masks.cpu().numpy()
    
    # Create subplot for each frame
    T = frames.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(T*4, 8))
    fig.suptitle(f'Sequence: {sequence_name}')
    
    for t in range(T):
        # Show frame
        axes[0, t].imshow(frames[t])
        axes[0, t].set_title(f'Frame {t}')
        axes[0, t].axis('off')
        
        # Show mask if available
        if masks is not None:
            axes[1, t].imshow(masks[t], cmap='tab20')
            axes[1, t].set_title(f'Mask {t}')
            axes[1, t].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f'{sequence_name}.png'
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_davis_dataset():
    # Set up transforms
    transform = VideoSequenceAugmentation(
        img_size=(480, 640),
        scale_range=(0.8, 1.2),
        rotation_range=(-10, 10),
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p_flip=0.5,
        normalize=True,
        train=True
    )
    
    # Create dataloader with proper WSL path
    davis_root = "/mnt/c/Datasets/DAVIS"  # WSL path
    
    # Verify DAVIS directory structure
    davis_path = Path(davis_root)
    print(f"Checking DAVIS directory structure at: {davis_path}")
    
    print("\nAvailable files in ImageSets:")
    imagesets_path = davis_path / 'ImageSets'
    if imagesets_path.exists():
        for path in imagesets_path.rglob('*'):
            print(f"Found: {path.relative_to(imagesets_path)}")
            
    print("\nTrying to load DAVIS 2017 dataset...")
    try:
        dataloader = build_davis_dataloader(
            root_path=davis_root,
            split='train',  # Changed from 'trainval' to 'train'
            batch_size=1,
            img_size=(480, 640),
            sequence_length=4,
            sequence_stride=2,
            transform=transform,
            year='2017'
        )
        
        print(f"\nDataset size: {len(dataloader)} sequences")
        
        # Create save directory for visualizations
        save_dir = Path("davis_visualization")
        save_dir.mkdir(exist_ok=True)
        print(f"\nVisualization will be saved to: {save_dir.absolute()}")
        
        # Visualize a few sequences
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Only show first 3 sequences
                break
                
            frames = batch['frames'].squeeze(0)  # Remove batch dimension
            masks = batch.get('masks')
            if masks is not None:
                masks = masks.squeeze(0)
            sequence = batch['sequence'][0]  # First item in batch
            
            print(f"\nSequence: {sequence}")
            print(f"Frames shape: {frames.shape}")
            if masks is not None:
                print(f"Masks shape: {masks.shape}")
            
            # Visualize
            visualize_sequence(frames, masks, sequence, save_dir)
            
            # Test without augmentation for comparison
            transform.train = False
            batch_no_aug = transform(batch)
            frames_no_aug = batch_no_aug['frames'].squeeze(0)
            masks_no_aug = batch_no_aug.get('masks')
            if masks_no_aug is not None:
                masks_no_aug = masks_no_aug.squeeze(0)
            
            visualize_sequence(frames_no_aug, masks_no_aug, 
                             f"{sequence}_no_aug", save_dir)
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease verify your DAVIS dataset has one of these files:")
        print("DAVIS/ImageSets/2017/train.txt")
        print("DAVIS/ImageSets/2017/val.txt")

if __name__ == "__main__":
    test_davis_dataset()