import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class DAVISDataset(Dataset):
    """DAVIS 2017 Dataset loading and preprocessing class."""
    
    def __init__(
        self,
        root_path: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (480, 640),
        sequence_length: int = 4,
        sequence_stride: int = 2,
        transform=None,
        year: str = '2017',
        specific_sequence: str = None  # Add this parameter
    ):
        """
        Initialize DAVIS dataset.
        
        Args:
            root_path: Path to DAVIS dataset root
            split: 'train', 'val', or 'test-dev'
            img_size: Target image size (height, width)
            sequence_length: Number of frames to load per sequence
            sequence_stride: Stride between frames in sequence
            transform: Optional transforms to apply
            year: DAVIS dataset year ('2017' or '2016')
            specific_sequence: Optional name of specific sequence to load
        """
        self.root_path = Path(root_path)
        self.split = split
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.transform = transform
        self.year = year
        self.specific_sequence = specific_sequence  # Store the parameter
        
        # Setup paths
        self.img_path = self.root_path / 'JPEGImages' / '480p'
        self.mask_path = self.root_path / 'Annotations' / '480p'
        
        # Load sequences
        self.sequences = self._load_sequences()
        self.frame_pairs = self._prepare_frame_pairs()
        
        print(f"\nDataset initialized:")
        print(f"- Number of sequences: {len(self.sequences)}")
        print(f"- Number of frame pairs: {len(self.frame_pairs)}")
        if specific_sequence:
            print(f"- Processing sequence: {specific_sequence}")
        
    def _load_sequences(self) -> List[str]:
        """Load sequence names based on split and optional specific sequence."""
        # Try different possible split file locations
        possible_paths = [
            self.root_path / 'ImageSets' / self.year / f'{self.split}.txt',
            self.root_path / 'ImageSets' / f'{self.split}.txt',
            self.root_path / 'ImageSets' / self.year / 'trainval.txt'
        ]
        
        split_file = None
        for path in possible_paths:
            if path.exists():
                split_file = path
                print(f"Found split file: {path}")
                break
        
        if split_file is None:
            raise FileNotFoundError(
                f"Could not find split file in any of these locations:\n"
                + "\n".join(str(p) for p in possible_paths)
            )
        
        with open(split_file, 'r') as f:
            sequences = [line.strip() for line in f.readlines()]
        
        # Filter for specific sequence if requested
        if self.specific_sequence is not None:
            if self.specific_sequence in sequences:
                sequences = [self.specific_sequence]
                print(f"Found requested sequence: {self.specific_sequence}")
            else:
                raise ValueError(
                    f"Sequence '{self.specific_sequence}' not found in split file. "
                    f"Available sequences: {sequences}"
                )
        
        return sequences
    
    def _prepare_frame_pairs(self) -> List[Tuple[str, List[str]]]:
        """Prepare frame pairs with temporal context."""
        frame_pairs = []
        
        for seq_name in self.sequences:
            # Get all frames for this sequence
            seq_path = self.img_path / seq_name
            if not seq_path.exists():
                print(f"Warning: Sequence path not found: {seq_path}")
                continue
                
            frames = sorted(list(seq_path.glob('*.jpg')))
            frame_names = [f.stem for f in frames]
            
            # Create sequences with stride
            for i in range(0, len(frame_names) - self.sequence_length + 1, self.sequence_stride):
                seq_frames = frame_names[i:i + self.sequence_length]
                if len(seq_frames) == self.sequence_length:
                    frame_pairs.append((seq_name, seq_frames))
                    
        return frame_pairs
    
    def _load_frame(self, seq_name: str, frame_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a single frame and its mask."""
        # Load image
        img_file = self.img_path / seq_name / f"{frame_name}.jpg"
        if not img_file.exists():
            raise FileNotFoundError(f"Image file not found: {img_file}")
            
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask if available (not available for test split)
        mask = None
        mask_file = self.mask_path / seq_name / f"{frame_name}.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Convert to torch tensors
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # [C, H, W]
        
        if mask is not None:
            mask = torch.from_numpy(mask).long()
        
        return img, mask
    
    def __len__(self) -> int:
        return len(self.frame_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of frames and masks."""
        seq_name, frame_names = self.frame_pairs[idx]
        
        # Load frames and masks
        frames = []
        masks = []
        
        for frame_name in frame_names:
            img, mask = self._load_frame(seq_name, frame_name)
            frames.append(img)
            if mask is not None:
                masks.append(mask)
        
        # Stack frames and masks
        frames = torch.stack(frames)  # [T, C, H, W]
        
        output = {
            'frames': frames,
            'sequence': seq_name,
            'frame_names': frame_names
        }
        
        if masks:
            masks = torch.stack(masks)  # [T, H, W]
            output['masks'] = masks
            
        if self.transform:
            output = self.transform(output)
            
        return output

def build_davis_dataloader(
    root_path: str,
    split: str = 'train',
    batch_size: int = 1,
    img_size: Tuple[int, int] = (480, 640),
    sequence_length: int = 4,
    sequence_stride: int = 2,
    num_workers: int = 4,
    transform=None,
    year: str = '2017',
    specific_sequence: str = None  # New parameter
) -> DataLoader:
    """Build DataLoader for DAVIS dataset."""
    dataset = DAVISDataset(
        root_path=root_path,
        split=split,
        img_size=img_size,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        transform=transform,
        year=year,
        specific_sequence=specific_sequence  # Pass to dataset
    )
    
    # Print dataset information
    print(f"\nDataset information:")
    print(f"- Total sequences: {len(dataset.sequences)}")
    if specific_sequence:
        print(f"- Processing sequence: {specific_sequence}")
    print(f"- Frame pairs: {len(dataset.frame_pairs)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader