# .gitignore

```
.Mask2Former
.checkpoints
./*.png
.codebase.md
```

# configs\binary_seg.yaml

```yaml
# Binary Video Segmentation Configuration
# configs/binary_seg.yaml

# Paths configuration
paths:
  davis_root: "/mnt/c/Datasets/DAVIS"  # Path to DAVIS dataset
  checkpoints: "checkpoints/binary_seg"  # Where to save model checkpoints
  logs: "logs/binary_seg"  # Where to save training logs
  visualizations: "visualizations/binary_seg"  # Where to save visualizations

# Model configuration
model:
  input_dim: 3  # RGB input
  hidden_dims: [32, 64, 128]  # Feature dimensions at each level
  d_state: 16  # State dimension for Mamba blocks
  temporal_window: 4  # Temporal context window size
  dropout: 0.1
  d_conv: 4  # Convolution dimension in Mamba
  expand: 2  # Expansion factor in Mamba

# Dataset configuration
dataset:
  batch_size: 4
  img_size: [384, 384]  # Height, Width
  sequence_length: 4  # Number of frames in each sequence
  sequence_stride: 1  # Stride between consecutive sequences
  num_workers: 4  # Number of data loading workers
  augmentation:
    scale_range: [0.5, 2.0]  # Instead of scale_min and scale_max
    rotation_range: [-10, 10]  # Instead of rotate_degrees
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p_flip: 0.5  # Instead of flip_prob

# Optimizer configuration
optimizer:
  type: "AdamW"
  lr: 0.0001
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler configuration
scheduler:
  type: "cosine"
  min_lr: 1.0e-6

# Loss function configuration
losses:
  ce_weight: 1.0  # Weight for binary cross-entropy loss
  dice_weight: 1.0  # Weight for dice loss
  temporal_weight: 0.5  # Weight for temporal consistency loss

# Training configuration
training:
  epochs: 100
  mixed_precision: true
  gradient_accumulation_steps: 4
  grad_clip_value: 1.0
  validate_every: 1  # Validate after every N epochs
  save_every: 10  # Save checkpoint every N epochs

# Visualization configuration
visualization:
  enabled: true
  dir: "visualizations/binary_seg"
  interval: 5  # Visualize every 5 validation runs

# Evaluation configuration
evaluation:
  enabled: true
  metrics: ["iou", "f1", "precision", "recall"]

# Optical flow configuration
flow:
  enabled: false  # Set to true if using optical flow
  method: "raft"  # Options: raft, farneback
  precomputed: false  # Set to true if using precomputed flows
  precomputed_path: null  # Path to precomputed flows (if applicable)
```

# configs\default.yaml

```yaml
                                                  # Model Configuration
model:
  input_dim: 3
  hidden_dims: [64, 128, 256]
  d_state: 32
  temporal_window: 8
  dropout: 0.1
  d_conv: 4
  expand: 2
  num_classes: 21
  # Add Mask2Former config inside model section
  mask2former:
    hidden_dim: 256
    num_queries: 100
    nheads: 8
    dim_feedforward: 2048
    dec_layers: 9
    mask_dim: 256
    enforce_input_project: false

# Dataset Configuration
# Dataset Configuration
dataset:
  img_size: [480, 640]
  sequence_length: 4  # Reduced from 8
  sequence_stride: 2
  batch_size: 1      # Reduced from 2
  num_workers: 2     # Reduced from 4
  augmentation:
    scale_range: [0.8, 1.2]
    rotation_range: [-10, 10]
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p_flip: 0.5

# Training Configuration
training:
  epochs: 100
  mixed_precision: true
  validate_every: 1
  save_every: 5
  gradient_accumulation_steps: 4  # Added this parameter

# Optimization Configuration
optimizer:
  type: 'AdamW'
  lr: 1.0e-4
  weight_decay: 0.01

# Learning Rate Schedule
scheduler:
  type: 'cosine'
  min_lr: 1e-6

# Loss weights
losses:
  ce_weight: 1.0
  dice_weight: 1.0
  temporal_weight: 1.0

# Paths

paths:
  davis_root: '/mnt/c/Datasets/DAVIS'
  checkpoints: 'checkpoints'
  logs: 'logs'
  visualizations: 'visualizations'
```

# configs\memory_efficient.yaml

```yaml
# Memory-efficient configuration for training on limited VRAM
# Model Configuration - reduced dimensions
model:
  input_dim: 3
  hidden_dims: [32, 64, 128]  # Reduced channel dimensions
  d_state: 16
  temporal_window: 4
  dropout: 0.1
  d_conv: 4
  expand: 2
  num_classes: 21
  mask2former:
    hidden_dim: 256          # Must be 256 to match Mask2Former expectations
    mask_dim: 256           # Must be 256 to match Mask2Former expectations
    num_queries: 50         # Reduced from 100 for memory efficiency
    nheads: 4              # Reduced from 8
    dim_feedforward: 512   # Reduced from 2048
    dec_layers: 6
    enforce_input_project: false

# Dataset Configuration
dataset:
  img_size: [240, 320]    # Reduced resolution
  sequence_length: 2      # Shorter sequences
  sequence_stride: 2
  batch_size: 1
  num_workers: 2
  augmentation:
    scale_range: [0.8, 1.2]
    rotation_range: [-10, 10]
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p_flip: 0.5

# Training Configuration
training:
  epochs: 100
  mixed_precision: true
  validate_every: 1
  save_every: 5
  gradient_accumulation_steps: 4


# Optimization Configuration
optimizer:
  type: 'AdamW'
  lr: 5.0e-5                      # Reduced from 1.0e-4
  weight_decay: 0.01

# Learning Rate Schedule
scheduler:
  type: 'cosine'
  min_lr: 1e-6

# Loss weights
losses:
  ce_weight: 1.0
  dice_weight: 1.0
  temporal_weight: 1.0

# Paths
paths:
  davis_root: '/mnt/c/Datasets/DAVIS'
  checkpoints: 'checkpoints'
  logs: 'logs'
  visualizations: 'visualizations'
```

# configs\optimized_rtx4070ti.yaml

```yaml
# optimized_rtx4070ti.yaml
model:
  input_dim: 3
  hidden_dims: [32, 64, 128]  # Slightly increased capacity
  d_state: 16
  temporal_window: 3         # Increased temporal window
  dropout: 0.2               # More dropout for regularization
  d_conv: 4
  expand: 2

dataset:
  img_size: [240, 320]
  sequence_length: 3         # Increased temporal context
  sequence_stride: 2
  batch_size: 4
  num_workers: 8
  augmentation:
    scale_range: [0.7, 1.3]  # More aggressive scaling
    rotation_range: [-15, 15]
    brightness: 0.3
    contrast: 0.3
    saturation: 0.3
    hue: 0.1
    p_flip: 0.5
    p_elastic: 0.3          # New elastic deformation

training:
  epochs: 100               # More epochs
  mixed_precision: true
  validate_every: 2
  save_every: 5
  gradient_accumulation_steps: 2  # Effective batch size of 8
  grad_clip_value: 0.5      # Moderate gradient clipping

optimizer:
  type: 'AdamW'
  lr: 6.0e-5                # Adjusted for OneCycleLR
  weight_decay: 0.005       # Increased regularization
  betas: [0.9, 0.99]        # Modified momentum parameters

scheduler:
  type: 'onecycle'          # New scheduler type
  min_lr: 1e-7

losses:
  ce_weight: 0.5            # Reduced CE weight
  dice_weight: 1.5          # Increased Dice weight
  boundary_weight: 1.0      # New boundary loss weight
  temporal_weight: 0.5

paths:
  davis_root: '/mnt/c/Datasets/DAVIS'
  checkpoints: 'checkpoints/rtx4070ti_optimized'
  logs: 'logs/rtx4070ti_optimized'
  visualizations: 'visualizations/rtx4070ti_optimized'

visualization:
  enabled: true
  interval: 10
  dir: 'visualizations/rtx4070ti_optimized'

evaluation:
  enabled: true
  metrics: ["iou", "f1", "boundary"]
```

# configs\rtx_3070_laptop.yaml

```yaml
# RTX 3070 Laptop GPU Configuration (8GB VRAM)
# Model Configuration
# Ultra memory-efficient configuration for RTX 3070 Laptop
model:
  input_dim: 3
  hidden_dims: [24, 48, 96]  # Significantly reduced dimensions
  d_state: 16
  temporal_window: 2         # Reduced temporal window
  dropout: 0.1
  d_conv: 4
  expand: 2
  num_instances: 12          # Fewer instances

dataset:
  img_size: [192, 256]       # Further reduced resolution
  sequence_length: 2         # Only process 2 frames at a time
  sequence_stride: 2
  batch_size: 1
  num_workers: 2
  augmentation:
    scale_range: [0.8, 1.2]
    rotation_range: [-10, 10]
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p_flip: 0.5

training:
  epochs: 100
  mixed_precision: true
  validate_every: 1
  save_every: 5
  gradient_accumulation_steps: 8  # Increased gradient accumulation
# Optimization Configuration
optimizer:
  type: 'AdamW'
  lr: 5.0e-5
  weight_decay: 0.01

# Learning Rate Schedule
scheduler:
  type: 'cosine'
  min_lr: 1e-6

# Loss weights
losses:
  ce_weight: 1.0
  dice_weight: 1.0
  temporal_weight: 1.0

# Paths
paths:
  davis_root: '/mnt/c/Datasets/DAVIS'
  checkpoints: 'checkpoints/rtx3070'
  logs: 'logs/rtx3070'
  visualizations: 'visualizations/rtx3070'

# Visualization and Evaluation
visualization:
  enabled: true
  dir: 'visualizations/rtx3070'
  interval: 5

evaluation:
  enabled: true
```

# configs\rtx_4070ti_super.yaml

```yaml
# Speed-optimized RTX 4070 Ti Super Configuration
model:
  input_dim: 3
  hidden_dims: [24, 48, 96]  # Slightly reduced dimensions for speed
  d_state: 16
  temporal_window: 2         # Reduced temporal window
  dropout: 0.1
  d_conv: 4
  expand: 2

dataset:
  img_size: [240, 320]       # Further reduced resolution for speed
  sequence_length: 2         # Minimal temporal context for faster processing
  sequence_stride: 2
  batch_size: 4              # Increased batch size for faster convergence
  num_workers: 8             # More workers for faster data loading
  augmentation:
    scale_range: [0.8, 1.2]
    rotation_range: [-10, 10]
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p_flip: 0.5

training:
  epochs: 50                # Fewer epochs
  mixed_precision: true
  validate_every: 2         # Validate less frequently
  save_every: 10
  grad_clip_value: 0.1     # Add this line to clip gradients and prevent NaNs
  gradient_accumulation_steps: 1  # No accumulation for direct updates

optimizer:
  type: 'AdamW'
  lr: 5.0e-5              # Lower learning rate (reduced by 4x)
  weight_decay: 0.01

scheduler:
  type: 'cosine'
  min_lr: 1e-6

losses:
  ce_weight: 1.0
  dice_weight: 1.0
  temporal_weight: 0.5

paths:
  davis_root: '/mnt/c/Datasets/DAVIS'
  checkpoints: 'checkpoints/rtx4070ti_super_fast'
  logs: 'logs/rtx4070ti_super_fast'
  visualizations: 'visualizations/rtx4070ti_super_fast'

visualization:
  enabled: true
  interval: 20              # Visualize much less frequently
  dir: 'visualizations/rtx4070ti_super_fast'

evaluation:
  enabled: true
  metrics: ["iou", "f1"]    # Fewer metrics to compute
```

# datasets\__init__.py

```py
# datasets/__init__.py

from .davis import DAVISDataset, build_davis_dataloader
from .transforms import VideoSequenceAugmentation

__all__ = [
    'DAVISDataset',
    'build_davis_dataloader',
    'VideoSequenceAugmentation'
]
```

# datasets\davis.py

```py
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
```

# datasets\transforms.py

```py
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter

class VideoSequenceAugmentation:
    """Augmentation for video sequences that maintains temporal consistency."""
    
    def __init__(
        self,
        img_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.7, 1.3),  # More aggressive scaling
        rotation_range: Tuple[int, int] = (-15, 15),    # Wider rotation
        brightness: float = 0.3,                        # More brightness variation
        contrast: float = 0.3,                          # More contrast variation
        saturation: float = 0.3,                        # More saturation variation
        hue: float = 0.1,
        p_flip: float = 0.5,
        p_elastic: float = 0.3,                         # Add elastic deformation
        elastic_alpha: float = 50,                      # Elastic deformation parameter
        elastic_sigma: float = 5,                       # Elastic deformation parameter
        p_cutout: float = 0.2,                          # Probability of applying cutout
        cutout_size: Tuple[float, float] = (0.1, 0.2),  # Cutout size as fraction of image
        normalize: bool = True,
        train: bool = True
    ):
        self.img_size = img_size
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p_flip = p_flip
        self.normalize = normalize
        self.train = train
        
        # Elastic deformation parameters
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        # Cutout parameters
        self.p_cutout = p_cutout
        self.cutout_size = cutout_size
        
        # ImageNet normalization stats
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def _get_params(self) -> Dict:
        """Get random transformation parameters."""
        params = {}
        
        if self.train:
            params['scale'] = random.uniform(*self.scale_range)
            params['angle'] = random.uniform(*self.rotation_range)
            params['brightness'] = random.uniform(max(0, 1-self.brightness), 1+self.brightness)
            params['contrast'] = random.uniform(max(0, 1-self.contrast), 1+self.contrast)
            params['saturation'] = random.uniform(max(0, 1-self.saturation), 1+self.saturation)
            params['hue'] = random.uniform(-self.hue, self.hue)
            params['flip'] = random.random() < self.p_flip
            
            # Elastic deformation parameters
            params['apply_elastic'] = random.random() < self.p_elastic
            if params['apply_elastic']:
                params['displacement'] = self._get_elastic_displacement()
                
            # Cutout parameters
            params['apply_cutout'] = random.random() < self.p_cutout
            if params['apply_cutout']:
                # Get random cutout size
                size_factor = random.uniform(*self.cutout_size)
                cutout_height = int(self.img_size[0] * size_factor)
                cutout_width = int(self.img_size[1] * size_factor)
                
                # Get random cutout position
                top = random.randint(0, self.img_size[0] - cutout_height)
                left = random.randint(0, self.img_size[1] - cutout_width)
                
                params['cutout'] = (top, left, cutout_height, cutout_width)
        
        return params
    
    def _get_elastic_displacement(self) -> torch.Tensor:
        """Create displacement fields for elastic deformation."""
        # Create random displacement fields
        dx = gaussian_filter(
            (np.random.rand(self.img_size[0], self.img_size[1]) * 2 - 1), 
            self.elastic_sigma
        ) * self.elastic_alpha
        
        dy = gaussian_filter(
            (np.random.rand(self.img_size[0], self.img_size[1]) * 2 - 1), 
            self.elastic_sigma
        ) * self.elastic_alpha
        
        # Convert to torch tensors
        return torch.from_numpy(np.array([dx, dy])).float()
    
    def _apply_elastic_transform(self, img: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation to an image."""
        # Ensure displacement is on the same device as the image
        if displacement.device != img.device:
            displacement = displacement.to(img.device)
        
        dx, dy = displacement
        h, w = img.shape[-2:]
        
        # Create meshgrid
        y, x = torch.meshgrid(torch.arange(h, device=img.device), 
                              torch.arange(w, device=img.device), 
                              indexing='ij')
        
        # Displace indices
        x_displaced = x.float() + dx
        y_displaced = y.float() + dy
        
        # Normalize to [-1, 1] for grid_sample
        x_norm = 2.0 * x_displaced / (w - 1) - 1.0
        y_norm = 2.0 * y_displaced / (h - 1) - 1.0
        
        # Create sampling grid
        grid = torch.stack([x_norm, y_norm], dim=-1)
        
        # Apply transformation using grid_sample
        # Need to add batch dimension for grid_sample
        if img.dim() == 3:  # [C, H, W]
            img_batch = img.unsqueeze(0)  # [1, C, H, W]
            out = F.grid_sample(
                img_batch, 
                grid.unsqueeze(0),  # [1, H, W, 2]
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )
            return out.squeeze(0)  # [C, H, W]
        else:  # Already has batch dimension
            return F.grid_sample(
                img, 
                grid.unsqueeze(0).expand(img.size(0), -1, -1, -1),
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )
    
    def _apply_cutout(self, img: torch.Tensor, params: Tuple[int, int, int, int]) -> torch.Tensor:
        """Apply cutout augmentation to an image."""
        top, left, height, width = params
        
        # Create a copy of the image
        img_cut = img.clone()
        
        if img.dim() == 3:  # [C, H, W]
            # Set the cutout region to zero (or other value)
            img_cut[:, top:top+height, left:left+width] = 0
        elif img.dim() == 4:  # [B, C, H, W]
            img_cut[:, :, top:top+height, left:left+width] = 0
        
        return img_cut
    
    def _apply_transform(
        self,
        frames: torch.Tensor,  # [T, C, H, W]
        masks: Optional[torch.Tensor],  # [T, H, W]
        params: Dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply transforms to frames and masks."""
        if not self.train:
            if self.normalize:
                frames = frames.sub(self.norm_mean).div(self.norm_std)
            return frames, masks
            
        T = frames.shape[0]  # Get temporal dimension
        frames = frames.clone()
        if masks is not None:
            masks = masks.clone()
        
        # Move to CPU for transforms
        frames = frames.cpu()
        if masks is not None:
            masks = masks.cpu()
        
        # Apply scaling and rotation consistently across sequence
        scale = params['scale']
        angle = params['angle']
        
        if scale != 1.0 or angle != 0:
            for t in range(T):
                # Transform frame
                frame = frames[t]
                frames[t] = TF.resize(
                    TF.rotate(frame, angle),
                    self.img_size,
                    antialias=True
                )
                
                # Transform mask if present
                if masks is not None:
                    mask = masks[t].unsqueeze(0)  # Add channel dim for transform
                    mask = TF.rotate(
                        mask, 
                        angle,
                        interpolation=TF.InterpolationMode.NEAREST
                    )
                    masks[t] = TF.resize(
                        mask,
                        self.img_size,
                        interpolation=TF.InterpolationMode.NEAREST
                    ).squeeze(0)
        
        # Apply elastic deformation if enabled
        if params.get('apply_elastic', False):
            displacement = params['displacement']
            for t in range(T):
                frames[t] = self._apply_elastic_transform(frames[t], displacement)
                if masks is not None:
                    # Need to handle mask differently as it's single-channel
                    mask_float = masks[t].float().unsqueeze(0)
                    mask_deformed = self._apply_elastic_transform(mask_float, displacement)
                    masks[t] = (mask_deformed.squeeze(0) > 0.5).long()
        
        # Apply cutout if enabled
        if params.get('apply_cutout', False):
            cutout_params = params['cutout']
            for t in range(T):
                frames[t] = self._apply_cutout(frames[t], cutout_params)
                # Optionally apply cutout to masks as well
                # if masks is not None:
                #     masks[t] = self._apply_cutout(masks[t].unsqueeze(0), cutout_params).squeeze(0)
        
        # Color jittering (apply to all frames consistently)
        frames = TF.adjust_brightness(frames, params['brightness'])
        frames = TF.adjust_contrast(frames, params['contrast'])
        frames = TF.adjust_saturation(frames, params['saturation'])
        frames = TF.adjust_hue(frames, params['hue'])
        
        # Horizontal flip
        if params['flip']:
            frames = TF.hflip(frames)
            if masks is not None:
                masks = TF.hflip(masks)
        
        # Normalize
        if self.normalize:
            frames = frames.sub(self.norm_mean).div(self.norm_std)
        
        return frames, masks
    
    def __call__(self, batch: Dict) -> Dict:
        """Apply transforms to a batch."""
        frames = batch['frames']  # [T, C, H, W]
        masks = batch.get('masks')  # [T, H, W] if present
        
        # Get transform parameters
        params = self._get_params()
        
        # Apply transforms
        frames, masks = self._apply_transform(frames, masks, params)
        
        # Update batch
        batch['frames'] = frames
        if masks is not None:
            batch['masks'] = masks
        
        return batch

# Example usage:
if __name__ == "__main__":
    transform = VideoSequenceAugmentation(
        img_size=(240, 320),
        scale_range=(0.7, 1.3),
        rotation_range=(-15, 15),
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
        p_flip=0.5,
        p_elastic=0.3,
        elastic_alpha=50,
        elastic_sigma=5, 
        p_cutout=0.2,
        cutout_size=(0.1, 0.2),
        normalize=True,
        train=True
    )
```

# evaluate.py

```py
import torch
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from models.model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from metrics.evaluator import DAVISEvaluator

def setup_logging(save_dir: Path):
    """Configure logging to both file and console for evaluation results."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )

# Replace the existing save_visualization function with your improved visualizer
def save_visualization(
    frames: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    sequence_name: str,
    save_dir: Path
):
    """
    Save visualization using the enhanced visualization tools.
    """
    from utils.visualization import VideoSegmentationVisualizer
    
    # Create visualizer with the specified save directory
    visualizer = VideoSegmentationVisualizer(save_dir=save_dir)
    
    # Create sequence visualization
    visualizer.visualize_sequence(
        frames=frames,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        sequence_name=sequence_name
    )
    
    # Create video visualization
    video_path = visualizer.create_video(
        frames=frames,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        sequence_name=sequence_name
    )
    logging.info(f"Created video visualization at {video_path}")
    
    # Create analysis dashboard
    visualizer.create_analysis_dashboard(
        frames=frames,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        sequence_name=sequence_name
    )

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    evaluator: DAVISEvaluator,
    device: torch.device,
    save_viz: bool = False,
    viz_dir: Optional[Path] = None
) -> dict:
    """
    Evaluate model on the dataset with progress tracking and comprehensive metrics.
    Returns a dictionary containing all computed metrics.
    """
    model.eval()
    all_metrics = []
    
    # Create progress bar for evaluation
    pbar = tqdm(dataloader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in pbar:
            # Move data to device
            frames = batch['frames'].to(device)
            masks = batch['masks'].to(device)
            sequence = batch['sequence'][0]  # Assuming batch_size=1
            
            # Forward pass
            outputs = model(frames)
            pred_masks = outputs['pred_masks']
            
            # Compute metrics for this sequence
            metrics = evaluator.evaluate_sequence(
                pred_masks[0],  # Remove batch dimension
                masks[0]
            )
            all_metrics.append(metrics)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'J_mean': f"{metrics['J_mean']:.4f}",
                'F_mean': f"{metrics.get('F_mean', 0):.4f}"
            })
            
            # Save visualizations if requested
            if save_viz and viz_dir is not None:
                save_visualization(
                    frames[0],
                    pred_masks[0],
                    masks[0],
                    sequence,
                    viz_dir
                )
    
    # Compute final metrics
    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = np.mean(values)
    
    return final_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate video segmentation model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization of predictions')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup paths and logging
    save_dir = Path(config['paths']['checkpoints'])
    setup_logging(save_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Build model and load checkpoint
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create data transform (no augmentation during evaluation)
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        normalize=True,
        train=False
    )
    
    # Create dataloader
    dataloader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split=args.split,
        batch_size=1,  # Use batch size 1 for evaluation
        transform=transform,
        **{k: v for k, v in config['dataset'].items() if k != 'batch_size'}
    )
    
    # Initialize evaluator
    evaluator = DAVISEvaluator()
    
    # Setup visualization directory if needed
    viz_dir = None
    if args.save_viz:
        viz_dir = Path(config['paths']['visualizations']) / args.split
        viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving visualizations to {viz_dir}")
    
    # Run evaluation
    logging.info(f"Evaluating model on {args.split} split...")
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        evaluator=evaluator,
        device=device,
        save_viz=args.save_viz,
        viz_dir=viz_dir
    )
    
    # Log results
    logging.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save metrics to file
    results_file = save_dir / f'metrics_{args.split}.txt'
    with open(results_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    logging.info(f"\nSaved metrics to {results_file}")

if __name__ == '__main__':
    main()
```

# logs\binary_seg\training_20250331_203508.log

```log
2025-03-31 20:35:08,060 - __main__ - INFO - Starting binary video segmentation training with config: configs/binary_seg.yaml
2025-03-31 20:35:08,064 - __main__ - INFO - Set random seed to 42
2025-03-31 20:35:08,302 - __main__ - INFO - Using device: cuda
2025-03-31 20:35:08,303 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:35:08,646 - __main__ - INFO - Model created: VideoMambaSegmentation

```

# logs\binary_seg\training_20250331_204255.log

```log
2025-03-31 20:42:55,990 - __main__ - INFO - Starting binary video segmentation training with config: configs/binary_seg.yaml
2025-03-31 20:42:55,994 - __main__ - INFO - Set random seed to 42
2025-03-31 20:42:56,242 - __main__ - INFO - Using device: cuda
2025-03-31 20:42:56,243 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:42:56,554 - __main__ - INFO - Model created: VideoMambaSegmentation

```

# logs\binary_seg\training_20250331_204541.log

```log
2025-03-31 20:45:41,840 - __main__ - INFO - Starting binary video segmentation training with config: configs/binary_seg.yaml
2025-03-31 20:45:41,843 - __main__ - INFO - Set random seed to 42
2025-03-31 20:45:42,086 - __main__ - INFO - Using device: cuda
2025-03-31 20:45:42,086 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:45:42,395 - __main__ - INFO - Model created: VideoMambaSegmentation

```

# logs\binary_seg\training_20250331_204658.log

```log
2025-03-31 20:46:58,490 - __main__ - INFO - Starting binary video segmentation training with config: configs/binary_seg.yaml
2025-03-31 20:46:58,494 - __main__ - INFO - Set random seed to 42
2025-03-31 20:46:58,739 - __main__ - INFO - Using device: cuda
2025-03-31 20:46:58,739 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:46:59,048 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-03-31 20:46:59,049 - __main__ - INFO - Created data augmentation with image size: [384, 384]
2025-03-31 20:46:59,049 - __main__ - INFO - Creating train data loader...
2025-03-31 20:46:59,463 - __main__ - INFO - Train loader created with 1008 batches
2025-03-31 20:46:59,463 - __main__ - INFO - Creating validation data loader...
2025-03-31 20:46:59,681 - __main__ - INFO - Validation loader created with 478 batches
2025-03-31 20:46:59,681 - __main__ - INFO - Creating optimizer: AdamW
2025-03-31 20:46:59,681 - __main__ - INFO - Creating scheduler: cosine

```

# logs\binary_seg\training_20250331_204802.log

```log
2025-03-31 20:48:02,161 - __main__ - INFO - Starting binary video segmentation training with config: configs/binary_seg.yaml
2025-03-31 20:48:02,165 - __main__ - INFO - Set random seed to 42
2025-03-31 20:48:02,415 - __main__ - INFO - Using device: cuda
2025-03-31 20:48:02,415 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:48:02,712 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-03-31 20:48:02,712 - __main__ - INFO - Created data augmentation with image size: [384, 384]
2025-03-31 20:48:02,713 - __main__ - INFO - Creating train data loader...
2025-03-31 20:48:03,097 - __main__ - INFO - Train loader created with 1008 batches
2025-03-31 20:48:03,097 - __main__ - INFO - Creating validation data loader...
2025-03-31 20:48:03,291 - __main__ - INFO - Validation loader created with 478 batches
2025-03-31 20:48:03,291 - __main__ - INFO - Creating optimizer: AdamW
2025-03-31 20:48:03,291 - __main__ - INFO - Creating scheduler: cosine
2025-03-31 20:48:03,292 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0

```

# logs\binary_seg\training_20250331_205400.log

```log
2025-03-31 20:54:00,517 - __main__ - INFO - Starting binary video segmentation training with config: configs/binary_seg.yaml
2025-03-31 20:54:00,520 - __main__ - INFO - Set random seed to 42
2025-03-31 20:54:00,767 - __main__ - INFO - Using device: cuda
2025-03-31 20:54:00,767 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:54:01,070 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-03-31 20:54:01,071 - __main__ - INFO - Created data augmentation with image size: [384, 384]
2025-03-31 20:54:01,071 - __main__ - INFO - Creating train data loader...
2025-03-31 20:54:01,386 - __main__ - INFO - Train loader created with 1008 batches
2025-03-31 20:54:01,387 - __main__ - INFO - Creating validation data loader...
2025-03-31 20:54:01,542 - __main__ - INFO - Validation loader created with 478 batches
2025-03-31 20:54:01,542 - __main__ - INFO - Creating optimizer: AdamW
2025-03-31 20:54:01,543 - __main__ - INFO - Creating scheduler: cosine
2025-03-31 20:54:01,543 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-03-31 20:54:01,549 - __main__ - INFO - Trainer initialized
2025-03-31 20:54:01,549 - __main__ - INFO - Starting training process...
2025-03-31 20:54:01,549 - utils.training - INFO - Starting training from epoch 0

```

# logs\rtx4070ti_optimized\training_20250401_174255.log

```log
2025-04-01 17:42:55,062 - __main__ - INFO - Starting binary video segmentation training with config: configs/optimized_rtx4070ti.yaml
2025-04-01 17:42:55,065 - __main__ - INFO - Set random seed to 42
2025-04-01 17:42:55,512 - __main__ - INFO - Using device: cuda
2025-04-01 17:42:55,512 - __main__ - INFO - Building binary segmentation model...
2025-04-01 17:42:56,337 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 17:42:56,338 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 17:42:56,338 - __main__ - INFO - Creating train data loader...
2025-04-01 17:42:56,755 - __main__ - INFO - Train loader created with 514 batches
2025-04-01 17:42:56,756 - __main__ - INFO - Creating validation data loader...
2025-04-01 17:42:56,960 - __main__ - INFO - Validation loader created with 244 batches
2025-04-01 17:42:56,960 - __main__ - INFO - Creating optimizer: AdamW

```

# logs\rtx4070ti_optimized\training_20250401_175831.log

```log
2025-04-01 17:58:31,714 - __main__ - INFO - Starting binary video segmentation training with config: configs/optimized_rtx4070ti.yaml
2025-04-01 17:58:31,718 - __main__ - INFO - Set random seed to 42
2025-04-01 17:58:31,967 - __main__ - INFO - Using device: cuda
2025-04-01 17:58:31,967 - __main__ - INFO - Building binary segmentation model...
2025-04-01 17:58:32,276 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 17:58:32,276 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 17:58:32,276 - __main__ - INFO - Creating train data loader...
2025-04-01 17:58:32,668 - __main__ - INFO - Train loader created with 514 batches
2025-04-01 17:58:32,668 - __main__ - INFO - Creating validation data loader...
2025-04-01 17:58:32,869 - __main__ - INFO - Validation loader created with 244 batches
2025-04-01 17:58:32,869 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 17:58:32,869 - __main__ - INFO - Creating scheduler: onecycle
2025-04-01 17:58:32,875 - __main__ - INFO - Trainer initialized
2025-04-01 17:58:32,875 - __main__ - INFO - Starting training process...
2025-04-01 17:58:32,875 - utils.training - INFO - Starting training from epoch 0

```

# logs\rtx4070ti_optimized\training_20250401_180003.log

```log
2025-04-01 18:00:03,350 - __main__ - INFO - Starting binary video segmentation training with config: configs/optimized_rtx4070ti.yaml
2025-04-01 18:00:03,354 - __main__ - INFO - Set random seed to 42
2025-04-01 18:00:03,607 - __main__ - INFO - Using device: cuda
2025-04-01 18:00:03,607 - __main__ - INFO - Building binary segmentation model...
2025-04-01 18:00:03,905 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 18:00:03,906 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 18:00:03,906 - __main__ - INFO - Creating train data loader...
2025-04-01 18:00:04,291 - __main__ - INFO - Train loader created with 514 batches
2025-04-01 18:00:04,291 - __main__ - INFO - Creating validation data loader...
2025-04-01 18:00:04,491 - __main__ - INFO - Validation loader created with 244 batches
2025-04-01 18:00:04,491 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 18:00:04,491 - __main__ - INFO - Creating scheduler: onecycle
2025-04-01 18:00:04,499 - __main__ - INFO - Trainer initialized
2025-04-01 18:00:04,499 - __main__ - INFO - Starting training process...
2025-04-01 18:00:04,499 - utils.training - INFO - Starting training from epoch 0

```

# logs\rtx4070ti_super_fast\training_20250331_205832.log

```log
2025-03-31 20:58:32,229 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-03-31 20:58:32,232 - __main__ - INFO - Set random seed to 42
2025-03-31 20:58:32,475 - __main__ - INFO - Using device: cuda
2025-03-31 20:58:32,475 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:58:32,727 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-03-31 20:58:32,728 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-03-31 20:58:32,728 - __main__ - INFO - Creating train data loader...
2025-03-31 20:58:33,113 - __main__ - INFO - Train loader created with 524 batches
2025-03-31 20:58:33,113 - __main__ - INFO - Creating validation data loader...
2025-03-31 20:58:33,368 - __main__ - INFO - Validation loader created with 249 batches
2025-03-31 20:58:33,368 - __main__ - INFO - Creating optimizer: AdamW
2025-03-31 20:58:33,369 - __main__ - INFO - Creating scheduler: cosine
2025-03-31 20:58:33,369 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-03-31 20:58:33,375 - __main__ - INFO - Trainer initialized
2025-03-31 20:58:33,375 - __main__ - INFO - Starting training process...
2025-03-31 20:58:33,375 - utils.training - INFO - Starting training from epoch 0

```

# logs\rtx4070ti_super_fast\training_20250401_123718.log

```log
2025-04-01 12:37:18,685 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-04-01 12:37:18,689 - __main__ - INFO - Set random seed to 42
2025-04-01 12:37:18,949 - __main__ - INFO - Using device: cuda
2025-04-01 12:37:18,950 - __main__ - INFO - Building binary segmentation model...
2025-04-01 12:37:19,509 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 12:37:19,509 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 12:37:19,510 - __main__ - INFO - Creating train data loader...
2025-04-01 12:37:19,915 - __main__ - INFO - Train loader created with 524 batches
2025-04-01 12:37:19,915 - __main__ - INFO - Creating validation data loader...
2025-04-01 12:37:20,183 - __main__ - INFO - Validation loader created with 249 batches
2025-04-01 12:37:20,183 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 12:37:20,184 - __main__ - INFO - Creating scheduler: cosine
2025-04-01 12:37:20,184 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-04-01 12:37:20,191 - __main__ - INFO - Trainer initialized
2025-04-01 12:37:20,192 - __main__ - INFO - Starting training process...
2025-04-01 12:37:20,192 - utils.training - INFO - Starting training from epoch 0
2025-04-01 12:40:17,899 - utils.training - INFO - Epoch 0 completed: Loss: nan, CE: nan, Dice: nan, LR: 0.000200

```

# logs\rtx4070ti_super_fast\training_20250401_124307.log

```log
2025-04-01 12:43:07,928 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-04-01 12:43:07,931 - __main__ - INFO - Set random seed to 42
2025-04-01 12:43:08,197 - __main__ - INFO - Using device: cuda
2025-04-01 12:43:08,197 - __main__ - INFO - Building binary segmentation model...
2025-04-01 12:43:08,624 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 12:43:08,625 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 12:43:08,625 - __main__ - INFO - Creating train data loader...
2025-04-01 12:43:09,014 - __main__ - INFO - Train loader created with 524 batches
2025-04-01 12:43:09,014 - __main__ - INFO - Creating validation data loader...
2025-04-01 12:43:09,268 - __main__ - INFO - Validation loader created with 249 batches
2025-04-01 12:43:09,268 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 12:43:09,268 - __main__ - INFO - Creating scheduler: cosine
2025-04-01 12:43:09,269 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-04-01 12:43:09,274 - __main__ - INFO - Trainer initialized
2025-04-01 12:43:09,274 - __main__ - INFO - Starting training process...
2025-04-01 12:43:09,275 - utils.training - INFO - Starting training from epoch 0
2025-04-01 12:46:05,399 - utils.training - INFO - Epoch 0 completed: Loss: nan, CE: nan, Dice: nan, LR: 0.000200

```

# logs\rtx4070ti_super_fast\training_20250401_125748.log

```log
2025-04-01 12:57:48,801 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-04-01 12:57:48,805 - __main__ - INFO - Set random seed to 42
2025-04-01 12:57:49,040 - __main__ - INFO - Using device: cuda
2025-04-01 12:57:49,040 - __main__ - INFO - Building binary segmentation model...
2025-04-01 12:57:49,352 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 12:57:49,353 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 12:57:49,353 - __main__ - INFO - Creating train data loader...
2025-04-01 12:57:49,724 - __main__ - INFO - Train loader created with 524 batches
2025-04-01 12:57:49,724 - __main__ - INFO - Creating validation data loader...
2025-04-01 12:57:49,976 - __main__ - INFO - Validation loader created with 249 batches
2025-04-01 12:57:49,977 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 12:57:49,977 - __main__ - INFO - Creating scheduler: cosine
2025-04-01 12:57:49,977 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-04-01 12:57:49,983 - __main__ - INFO - Trainer initialized
2025-04-01 12:57:49,983 - __main__ - INFO - Starting training process...
2025-04-01 12:57:49,983 - utils.training - INFO - Starting training from epoch 0
2025-04-01 13:00:50,443 - utils.training - INFO - Epoch 0 completed: Loss: nan, CE: nan, Dice: nan, LR: 0.000200
2025-04-01 13:00:50,444 - utils.training - INFO - Epoch 0 training loss: nan
2025-04-01 13:03:48,124 - utils.training - INFO - Epoch 0 completed: Loss: nan, CE: nan, Dice: nan, LR: 0.000198
2025-04-01 13:03:48,125 - utils.training - INFO - Epoch 1 training loss: nan

```

# logs\rtx4070ti_super_fast\training_20250401_130429.log

```log
2025-04-01 13:04:29,344 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-04-01 13:04:29,346 - __main__ - INFO - Set random seed to 42
2025-04-01 13:04:29,589 - __main__ - INFO - Using device: cuda
2025-04-01 13:04:29,589 - __main__ - INFO - Building binary segmentation model...
2025-04-01 13:04:29,831 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 13:04:29,831 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 13:04:29,832 - __main__ - INFO - Creating train data loader...
2025-04-01 13:04:30,203 - __main__ - INFO - Train loader created with 524 batches
2025-04-01 13:04:30,203 - __main__ - INFO - Creating validation data loader...
2025-04-01 13:04:30,454 - __main__ - INFO - Validation loader created with 249 batches
2025-04-01 13:04:30,454 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 13:04:30,454 - __main__ - INFO - Creating scheduler: cosine
2025-04-01 13:04:30,455 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-04-01 13:04:30,460 - __main__ - INFO - Trainer initialized
2025-04-01 13:04:30,460 - __main__ - INFO - Starting training process...
2025-04-01 13:04:30,460 - utils.training - INFO - Starting training from epoch 0
2025-04-01 13:07:27,954 - utils.training - INFO - Epoch 0 completed: Loss: 1.5318, CE: 0.7215, Dice: 0.8103, LR: 0.000050
2025-04-01 13:07:27,954 - utils.training - INFO - Epoch 0 training loss: 1.5318
2025-04-01 13:10:23,540 - utils.training - INFO - Epoch 0 completed: Loss: 1.5105, CE: 0.6996, Dice: 0.8109, LR: 0.000050
2025-04-01 13:10:23,540 - utils.training - INFO - Epoch 1 training loss: 1.5105

```

# logs\rtx4070ti_super_fast\training_20250401_131538.log

```log
2025-04-01 13:15:38,062 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-04-01 13:15:38,064 - __main__ - INFO - Set random seed to 42
2025-04-01 13:15:38,306 - __main__ - INFO - Using device: cuda
2025-04-01 13:15:38,306 - __main__ - INFO - Building binary segmentation model...
2025-04-01 13:15:38,535 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 13:15:38,536 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 13:15:38,536 - __main__ - INFO - Creating train data loader...
2025-04-01 13:15:38,902 - __main__ - INFO - Train loader created with 524 batches
2025-04-01 13:15:38,902 - __main__ - INFO - Creating validation data loader...
2025-04-01 13:15:39,149 - __main__ - INFO - Validation loader created with 249 batches
2025-04-01 13:15:39,149 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 13:15:39,150 - __main__ - INFO - Creating scheduler: cosine
2025-04-01 13:15:39,151 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-04-01 13:15:39,156 - __main__ - INFO - Trainer initialized
2025-04-01 13:15:39,156 - __main__ - INFO - Starting training process...
2025-04-01 13:15:39,156 - utils.training - INFO - Starting training from epoch 0
2025-04-01 13:18:37,559 - utils.training - INFO - Epoch 0 completed: Loss: 1.5318, CE: 0.7215, Dice: 0.8103, LR: 0.000050
2025-04-01 13:21:34,338 - utils.training - INFO - Epoch 0 completed: Loss: 1.5105, CE: 0.6996, Dice: 0.8109, LR: 0.000050
2025-04-01 13:21:35,070 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399238].
2025-04-01 13:21:35,072 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 13:21:35,079 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25597].
2025-04-01 13:21:35,259 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399238].
2025-04-01 13:21:35,261 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 13:21:35,264 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25597].
2025-04-01 13:21:40,398 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:21:40,398 - utils.training - INFO - Global: J&F: 0.0887, J_mean: 0.1308, T_mean: 0.9903
2025-04-01 13:21:45,761 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:21:45,761 - utils.training - INFO - Global: J&F: 0.1000, J_mean: 0.1609, T_mean: 0.9921
2025-04-01 13:21:48,623 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4291956].
2025-04-01 13:21:48,628 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.095132..108.97168].
2025-04-01 13:21:48,631 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0951612..108.95526].
2025-04-01 13:21:48,789 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0891676..2.419701].
2025-04-01 13:21:48,792 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0691736..108.96788].
2025-04-01 13:21:48,795 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0891676..108.95332].
2025-04-01 13:21:51,425 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:21:51,425 - utils.training - INFO - Global: J&F: 0.0827, J_mean: 0.1427, T_mean: 0.9960
2025-04-01 13:21:56,753 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:21:56,754 - utils.training - INFO - Global: J&F: 0.0784, J_mean: 0.1292, T_mean: 0.9962
2025-04-01 13:22:02,007 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:02,008 - utils.training - INFO - Global: J&F: 0.0728, J_mean: 0.1105, T_mean: 0.9977
2025-04-01 13:22:02,320 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399555].
2025-04-01 13:22:02,322 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 13:22:02,325 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.61107].
2025-04-01 13:22:02,465 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399555].
2025-04-01 13:22:02,467 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 13:22:02,470 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.62651].
2025-04-01 13:22:07,500 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:07,500 - utils.training - INFO - Global: J&F: 0.1247, J_mean: 0.2300, T_mean: 0.9979
2025-04-01 13:22:12,770 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:12,771 - utils.training - INFO - Global: J&F: 0.0853, J_mean: 0.1370, T_mean: 0.9952
2025-04-01 13:22:15,631 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3141396].
2025-04-01 13:22:15,634 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92566].
2025-04-01 13:22:15,637 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.10182].
2025-04-01 13:22:15,803 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3303018].
2025-04-01 13:22:15,805 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.921646].
2025-04-01 13:22:15,808 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.04208].
2025-04-01 13:22:18,314 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:18,314 - utils.training - INFO - Global: J&F: 0.0769, J_mean: 0.1275, T_mean: 0.9877
2025-04-01 13:22:23,603 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:23,604 - utils.training - INFO - Global: J&F: 0.0757, J_mean: 0.1219, T_mean: 0.9975
2025-04-01 13:22:28,855 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:28,855 - utils.training - INFO - Global: J&F: 0.1155, J_mean: 0.1775, T_mean: 0.9992
2025-04-01 13:22:29,167 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0068977..1.6809332].
2025-04-01 13:22:29,171 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0068977..108.67237].
2025-04-01 13:22:29,173 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0068977..153.40898].
2025-04-01 13:22:29,345 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0069392..1.6843803].
2025-04-01 13:22:29,348 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0069392..108.67375].
2025-04-01 13:22:29,352 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0069392..153.48389].
2025-04-01 13:22:34,477 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:34,477 - utils.training - INFO - Global: J&F: 0.1535, J_mean: 0.2616, T_mean: 0.9988
2025-04-01 13:22:39,792 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:39,792 - utils.training - INFO - Global: J&F: 0.1314, J_mean: 0.2248, T_mean: 0.9935
2025-04-01 13:22:42,099 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:22:42,099 - utils.training - INFO - Global: J&F: 0.1016, J_mean: 0.1595, T_mean: 0.9984
2025-04-01 13:22:42,170 - utils.training - INFO - Epoch 1 validation: val_loss: 1.6690 J&F: 0.1016 J_mean: 0.1595 F_mean: 0.0438 T_mean: 0.9984 instance_stability: 1.0000
2025-04-01 13:22:42,339 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 13:25:37,751 - utils.training - INFO - Epoch 0 completed: Loss: 1.5067, CE: 0.6977, Dice: 0.8089, LR: 0.000049
2025-04-01 13:28:33,507 - utils.training - INFO - Epoch 0 completed: Loss: 1.5025, CE: 0.6984, Dice: 0.8041, LR: 0.000048
2025-04-01 13:28:34,100 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8350892..2.1417532].
2025-04-01 13:28:34,104 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8350892..108.85228].
2025-04-01 13:28:34,110 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8350892..140.0567].
2025-04-01 13:28:34,323 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8346401..2.1412444].
2025-04-01 13:28:34,325 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8346401..108.85211].
2025-04-01 13:28:34,328 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8346401..140.0565].
2025-04-01 13:28:39,448 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:28:39,448 - utils.training - INFO - Global: J&F: 0.0927, J_mean: 0.1424, T_mean: 0.9914
2025-04-01 13:28:44,767 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:28:44,767 - utils.training - INFO - Global: J&F: 0.0956, J_mean: 0.1590, T_mean: 0.9936
2025-04-01 13:28:47,628 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3138883].
2025-04-01 13:28:47,632 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92555].
2025-04-01 13:28:47,635 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92555].
2025-04-01 13:28:47,777 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.314261].
2025-04-01 13:28:47,780 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.925705].
2025-04-01 13:28:47,783 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.925476].
2025-04-01 13:28:50,370 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:28:50,370 - utils.training - INFO - Global: J&F: 0.0936, J_mean: 0.1608, T_mean: 0.9958
2025-04-01 13:28:55,711 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:28:55,712 - utils.training - INFO - Global: J&F: 0.1249, J_mean: 0.2068, T_mean: 0.9934
2025-04-01 13:29:01,062 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:01,062 - utils.training - INFO - Global: J&F: 0.1089, J_mean: 0.1873, T_mean: 0.9949
2025-04-01 13:29:01,375 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8206496..2.304155].
2025-04-01 13:29:01,378 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8206496..108.92166].
2025-04-01 13:29:01,381 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8206496..153.44014].
2025-04-01 13:29:01,524 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8198638..2.304955].
2025-04-01 13:29:01,525 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8198638..108.92198].
2025-04-01 13:29:01,528 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8198638..153.45287].
2025-04-01 13:29:06,657 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:06,658 - utils.training - INFO - Global: J&F: 0.1205, J_mean: 0.2176, T_mean: 0.9965
2025-04-01 13:29:12,149 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:12,149 - utils.training - INFO - Global: J&F: 0.1013, J_mean: 0.1662, T_mean: 0.9948
2025-04-01 13:29:15,029 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 13:29:15,031 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 13:29:15,034 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 13:29:15,172 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 13:29:15,174 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 13:29:15,177 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25209].
2025-04-01 13:29:17,719 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:17,719 - utils.training - INFO - Global: J&F: 0.0927, J_mean: 0.1567, T_mean: 0.9927
2025-04-01 13:29:23,125 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:23,126 - utils.training - INFO - Global: J&F: 0.1042, J_mean: 0.1751, T_mean: 0.9933
2025-04-01 13:29:28,490 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:28,490 - utils.training - INFO - Global: J&F: 0.1351, J_mean: 0.2192, T_mean: 0.9953
2025-04-01 13:29:28,801 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399803].
2025-04-01 13:29:28,804 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 13:29:28,807 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89955].
2025-04-01 13:29:28,976 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399803].
2025-04-01 13:29:28,978 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 13:29:28,981 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89955].
2025-04-01 13:29:34,175 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:34,175 - utils.training - INFO - Global: J&F: 0.1129, J_mean: 0.1858, T_mean: 0.9961
2025-04-01 13:29:39,505 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:39,505 - utils.training - INFO - Global: J&F: 0.1021, J_mean: 0.1592, T_mean: 0.9944
2025-04-01 13:29:41,864 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:29:41,865 - utils.training - INFO - Global: J&F: 0.1260, J_mean: 0.1985, T_mean: 0.9892
2025-04-01 13:29:41,940 - utils.training - INFO - Epoch 3 validation: val_loss: 1.5415 J&F: 0.1260 J_mean: 0.1985 F_mean: 0.0535 T_mean: 0.9892 instance_stability: 1.0000
2025-04-01 13:29:42,111 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 13:32:37,508 - utils.training - INFO - Epoch 0 completed: Loss: 1.4978, CE: 0.6977, Dice: 0.8001, LR: 0.000046
2025-04-01 13:35:33,271 - utils.training - INFO - Epoch 0 completed: Loss: 1.4927, CE: 0.6972, Dice: 0.7955, LR: 0.000044
2025-04-01 13:35:33,871 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.007936..2.0534825].
2025-04-01 13:35:33,876 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.007936..108.81627].
2025-04-01 13:35:33,881 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.007936..139.96887].
2025-04-01 13:35:34,062 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0077534..2.0548885].
2025-04-01 13:35:34,064 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0077534..108.816925].
2025-04-01 13:35:34,067 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0077534..139.97456].
2025-04-01 13:35:39,286 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:35:39,286 - utils.training - INFO - Global: J&F: 0.1735, J_mean: 0.2908, T_mean: 0.9883
2025-04-01 13:35:44,801 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:35:44,801 - utils.training - INFO - Global: J&F: 0.0890, J_mean: 0.1445, T_mean: 0.9916
2025-04-01 13:35:47,719 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9211053..2.1625993].
2025-04-01 13:35:47,722 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9211053..108.865036].
2025-04-01 13:35:47,725 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9087244..108.862366].
2025-04-01 13:35:47,902 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9066013..2.160487].
2025-04-01 13:35:47,905 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9066013..108.8642].
2025-04-01 13:35:47,908 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9066013..108.86235].
2025-04-01 13:35:50,536 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:35:50,536 - utils.training - INFO - Global: J&F: 0.1021, J_mean: 0.1780, T_mean: 0.9952
2025-04-01 13:35:55,846 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:35:55,847 - utils.training - INFO - Global: J&F: 0.1063, J_mean: 0.1722, T_mean: 0.9912
2025-04-01 13:36:01,282 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:01,283 - utils.training - INFO - Global: J&F: 0.1212, J_mean: 0.2029, T_mean: 0.9892
2025-04-01 13:36:01,619 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0687451..1.917734].
2025-04-01 13:36:01,620 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0687451..108.55557].
2025-04-01 13:36:01,624 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0686605..153.25804].
2025-04-01 13:36:01,771 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0686219..1.9031312].
2025-04-01 13:36:01,772 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0685372..108.54971].
2025-04-01 13:36:01,775 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0685372..153.26964].
2025-04-01 13:36:06,879 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:06,880 - utils.training - INFO - Global: J&F: 0.1580, J_mean: 0.2848, T_mean: 0.9936
2025-04-01 13:36:12,257 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:12,258 - utils.training - INFO - Global: J&F: 0.0930, J_mean: 0.1523, T_mean: 0.9935
2025-04-01 13:36:15,161 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.949214..2.3596659].
2025-04-01 13:36:15,164 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.949214..108.94081].
2025-04-01 13:36:15,168 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.949214..140.1131].
2025-04-01 13:36:15,304 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9497089..2.3601756].
2025-04-01 13:36:15,306 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9497089..108.93564].
2025-04-01 13:36:15,309 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9497089..140.07967].
2025-04-01 13:36:17,893 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:17,893 - utils.training - INFO - Global: J&F: 0.0987, J_mean: 0.1579, T_mean: 0.9913
2025-04-01 13:36:23,361 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:23,361 - utils.training - INFO - Global: J&F: 0.1369, J_mean: 0.2307, T_mean: 0.9873
2025-04-01 13:36:28,828 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:28,829 - utils.training - INFO - Global: J&F: 0.1633, J_mean: 0.2584, T_mean: 0.9871
2025-04-01 13:36:29,144 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0763974..2.027336].
2025-04-01 13:36:29,147 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0763974..108.810936].
2025-04-01 13:36:29,150 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0763974..153.52676].
2025-04-01 13:36:29,323 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0764034..2.0239816].
2025-04-01 13:36:29,324 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0764034..108.80959].
2025-04-01 13:36:29,327 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0764034..153.60588].
2025-04-01 13:36:34,597 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:34,598 - utils.training - INFO - Global: J&F: 0.1573, J_mean: 0.2650, T_mean: 0.9918
2025-04-01 13:36:40,050 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:40,050 - utils.training - INFO - Global: J&F: 0.1362, J_mean: 0.2206, T_mean: 0.9857
2025-04-01 13:36:42,456 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:36:42,456 - utils.training - INFO - Global: J&F: 0.1357, J_mean: 0.2239, T_mean: 0.9839
2025-04-01 13:36:42,536 - utils.training - INFO - Epoch 5 validation: val_loss: 1.3610 J&F: 0.1357 J_mean: 0.2239 F_mean: 0.0476 T_mean: 0.9839 instance_stability: 1.0000
2025-04-01 13:36:42,701 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast

```

# logs\rtx4070ti_super_fast\training_20250401_133703.log

```log
2025-04-01 13:37:03,041 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-04-01 13:37:03,043 - __main__ - INFO - Set random seed to 42
2025-04-01 13:37:03,282 - __main__ - INFO - Using device: cuda
2025-04-01 13:37:03,282 - __main__ - INFO - Building binary segmentation model...
2025-04-01 13:37:03,500 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-04-01 13:37:03,500 - __main__ - INFO - Created data augmentation with image size: [240, 320]
2025-04-01 13:37:03,501 - __main__ - INFO - Creating train data loader...
2025-04-01 13:37:03,871 - __main__ - INFO - Train loader created with 524 batches
2025-04-01 13:37:03,871 - __main__ - INFO - Creating validation data loader...
2025-04-01 13:37:04,118 - __main__ - INFO - Validation loader created with 249 batches
2025-04-01 13:37:04,119 - __main__ - INFO - Creating optimizer: AdamW
2025-04-01 13:37:04,119 - __main__ - INFO - Creating scheduler: cosine
2025-04-01 13:37:04,120 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-04-01 13:37:04,124 - __main__ - INFO - Trainer initialized
2025-04-01 13:37:04,125 - __main__ - INFO - Starting training process...
2025-04-01 13:37:04,125 - utils.training - INFO - Starting training from epoch 0
2025-04-01 13:40:00,792 - utils.training - INFO - Epoch 0 completed: Loss: 1.5318, CE: 0.7215, Dice: 0.8103, LR: 0.000050
2025-04-01 13:42:58,031 - utils.training - INFO - Epoch 1 completed: Loss: 1.5105, CE: 0.6996, Dice: 0.8109, LR: 0.000050
2025-04-01 13:42:58,677 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399238].
2025-04-01 13:42:58,681 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 13:42:58,686 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25597].
2025-04-01 13:42:58,939 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399238].
2025-04-01 13:42:58,941 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 13:42:58,945 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25597].
2025-04-01 13:43:04,357 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:04,357 - utils.training - INFO - Global: J&F: 0.0903, J_mean: 0.1342, T_mean: 0.9903
2025-04-01 13:43:09,770 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:09,771 - utils.training - INFO - Global: J&F: 0.1003, J_mean: 0.1619, T_mean: 0.9921
2025-04-01 13:43:12,680 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4291956].
2025-04-01 13:43:12,683 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.095132..108.97168].
2025-04-01 13:43:12,686 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0951612..108.95526].
2025-04-01 13:43:12,867 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0891676..2.419701].
2025-04-01 13:43:12,870 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0691736..108.96788].
2025-04-01 13:43:12,873 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0891676..108.95332].
2025-04-01 13:43:15,555 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:15,555 - utils.training - INFO - Global: J&F: 0.0835, J_mean: 0.1444, T_mean: 0.9959
2025-04-01 13:43:20,942 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:20,943 - utils.training - INFO - Global: J&F: 0.0774, J_mean: 0.1275, T_mean: 0.9962
2025-04-01 13:43:26,240 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:26,240 - utils.training - INFO - Global: J&F: 0.0735, J_mean: 0.1120, T_mean: 0.9977
2025-04-01 13:43:26,556 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399555].
2025-04-01 13:43:26,559 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 13:43:26,562 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.61107].
2025-04-01 13:43:26,696 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399555].
2025-04-01 13:43:26,698 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 13:43:26,701 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.62651].
2025-04-01 13:43:31,822 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:31,823 - utils.training - INFO - Global: J&F: 0.1231, J_mean: 0.2270, T_mean: 0.9979
2025-04-01 13:43:37,164 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:37,164 - utils.training - INFO - Global: J&F: 0.0859, J_mean: 0.1384, T_mean: 0.9954
2025-04-01 13:43:40,073 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3141396].
2025-04-01 13:43:40,076 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92566].
2025-04-01 13:43:40,079 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.10182].
2025-04-01 13:43:40,217 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3303018].
2025-04-01 13:43:40,219 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.921646].
2025-04-01 13:43:40,222 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.04208].
2025-04-01 13:43:42,745 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:42,745 - utils.training - INFO - Global: J&F: 0.0761, J_mean: 0.1253, T_mean: 0.9877
2025-04-01 13:43:48,125 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:48,125 - utils.training - INFO - Global: J&F: 0.0770, J_mean: 0.1241, T_mean: 0.9976
2025-04-01 13:43:53,432 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:53,432 - utils.training - INFO - Global: J&F: 0.1149, J_mean: 0.1754, T_mean: 0.9992
2025-04-01 13:43:53,748 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0068977..1.6809332].
2025-04-01 13:43:53,753 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0068977..108.67237].
2025-04-01 13:43:53,756 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0068977..153.40898].
2025-04-01 13:43:53,930 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0069392..1.6843803].
2025-04-01 13:43:53,934 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0069392..108.67375].
2025-04-01 13:43:53,937 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0069392..153.48389].
2025-04-01 13:43:59,133 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:43:59,134 - utils.training - INFO - Global: J&F: 0.1418, J_mean: 0.2425, T_mean: 0.9988
2025-04-01 13:44:04,538 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:44:04,538 - utils.training - INFO - Global: J&F: 0.1312, J_mean: 0.2251, T_mean: 0.9936
2025-04-01 13:44:06,884 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:44:06,884 - utils.training - INFO - Global: J&F: 0.1002, J_mean: 0.1562, T_mean: 0.9984
2025-04-01 13:44:06,961 - utils.training - INFO - Epoch 1 validation: val_loss: 1.6693 J&F: 0.1002 J_mean: 0.1562 F_mean: 0.0441 T_mean: 0.9984 instance_stability: 1.0000
2025-04-01 13:44:07,109 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 13:47:07,050 - utils.training - INFO - Epoch 2 completed: Loss: 1.5067, CE: 0.6977, Dice: 0.8090, LR: 0.000049
2025-04-01 13:50:06,440 - utils.training - INFO - Epoch 3 completed: Loss: 1.5025, CE: 0.6984, Dice: 0.8041, LR: 0.000048
2025-04-01 13:50:06,998 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8350892..2.1417532].
2025-04-01 13:50:07,001 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8350892..108.85228].
2025-04-01 13:50:07,005 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8350892..140.0567].
2025-04-01 13:50:07,232 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8346401..2.1412444].
2025-04-01 13:50:07,234 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8346401..108.85211].
2025-04-01 13:50:07,237 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8346401..140.0565].
2025-04-01 13:50:12,420 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:12,420 - utils.training - INFO - Global: J&F: 0.0933, J_mean: 0.1429, T_mean: 0.9915
2025-04-01 13:50:17,851 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:17,851 - utils.training - INFO - Global: J&F: 0.0898, J_mean: 0.1491, T_mean: 0.9936
2025-04-01 13:50:20,752 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3138883].
2025-04-01 13:50:20,756 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92555].
2025-04-01 13:50:20,759 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92555].
2025-04-01 13:50:20,918 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.314261].
2025-04-01 13:50:20,921 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.925705].
2025-04-01 13:50:20,924 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.925476].
2025-04-01 13:50:23,545 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:23,546 - utils.training - INFO - Global: J&F: 0.0917, J_mean: 0.1571, T_mean: 0.9959
2025-04-01 13:50:28,954 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:28,955 - utils.training - INFO - Global: J&F: 0.1248, J_mean: 0.2068, T_mean: 0.9935
2025-04-01 13:50:34,360 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:34,360 - utils.training - INFO - Global: J&F: 0.1128, J_mean: 0.1946, T_mean: 0.9951
2025-04-01 13:50:34,680 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8206496..2.304155].
2025-04-01 13:50:34,681 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8206496..108.92166].
2025-04-01 13:50:34,685 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8206496..153.44014].
2025-04-01 13:50:34,831 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8198638..2.304955].
2025-04-01 13:50:34,832 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8198638..108.92198].
2025-04-01 13:50:34,835 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8198638..153.45287].
2025-04-01 13:50:39,962 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:39,962 - utils.training - INFO - Global: J&F: 0.1215, J_mean: 0.2196, T_mean: 0.9966
2025-04-01 13:50:45,339 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:45,339 - utils.training - INFO - Global: J&F: 0.1135, J_mean: 0.1863, T_mean: 0.9949
2025-04-01 13:50:48,245 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 13:50:48,247 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 13:50:48,250 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 13:50:48,388 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 13:50:48,389 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 13:50:48,393 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25209].
2025-04-01 13:50:50,961 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:50,961 - utils.training - INFO - Global: J&F: 0.0934, J_mean: 0.1577, T_mean: 0.9929
2025-04-01 13:50:56,415 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:50:56,415 - utils.training - INFO - Global: J&F: 0.1085, J_mean: 0.1822, T_mean: 0.9935
2025-04-01 13:51:01,859 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:51:01,860 - utils.training - INFO - Global: J&F: 0.1366, J_mean: 0.2213, T_mean: 0.9955
2025-04-01 13:51:02,205 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399803].
2025-04-01 13:51:02,208 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 13:51:02,211 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89955].
2025-04-01 13:51:02,367 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399803].
2025-04-01 13:51:02,369 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 13:51:02,372 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89955].
2025-04-01 13:51:07,637 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:51:07,637 - utils.training - INFO - Global: J&F: 0.1146, J_mean: 0.1897, T_mean: 0.9963
2025-04-01 13:51:12,998 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:51:12,999 - utils.training - INFO - Global: J&F: 0.1010, J_mean: 0.1571, T_mean: 0.9946
2025-04-01 13:51:15,397 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:51:15,398 - utils.training - INFO - Global: J&F: 0.1222, J_mean: 0.1919, T_mean: 0.9894
2025-04-01 13:51:15,478 - utils.training - INFO - Epoch 3 validation: val_loss: 1.5448 J&F: 0.1222 J_mean: 0.1919 F_mean: 0.0524 T_mean: 0.9894 instance_stability: 1.0000
2025-04-01 13:51:15,647 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 13:54:15,328 - utils.training - INFO - Epoch 4 completed: Loss: 1.4979, CE: 0.6977, Dice: 0.8002, LR: 0.000046
2025-04-01 13:57:14,922 - utils.training - INFO - Epoch 5 completed: Loss: 1.4928, CE: 0.6972, Dice: 0.7956, LR: 0.000044
2025-04-01 13:57:15,533 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.007936..2.0534825].
2025-04-01 13:57:15,536 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.007936..108.81627].
2025-04-01 13:57:15,541 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.007936..139.96887].
2025-04-01 13:57:15,754 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0077534..2.0548885].
2025-04-01 13:57:15,757 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0077534..108.816925].
2025-04-01 13:57:15,760 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0077534..139.97456].
2025-04-01 13:57:20,945 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:20,945 - utils.training - INFO - Global: J&F: 0.1739, J_mean: 0.2914, T_mean: 0.9881
2025-04-01 13:57:26,345 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:26,345 - utils.training - INFO - Global: J&F: 0.0930, J_mean: 0.1523, T_mean: 0.9915
2025-04-01 13:57:29,256 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9211053..2.1625993].
2025-04-01 13:57:29,258 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9211053..108.865036].
2025-04-01 13:57:29,262 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9087244..108.862366].
2025-04-01 13:57:29,441 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9066013..2.160487].
2025-04-01 13:57:29,443 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9066013..108.8642].
2025-04-01 13:57:29,446 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9066013..108.86235].
2025-04-01 13:57:32,086 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:32,086 - utils.training - INFO - Global: J&F: 0.1039, J_mean: 0.1810, T_mean: 0.9951
2025-04-01 13:57:37,447 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:37,447 - utils.training - INFO - Global: J&F: 0.1065, J_mean: 0.1727, T_mean: 0.9910
2025-04-01 13:57:42,921 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:42,922 - utils.training - INFO - Global: J&F: 0.1252, J_mean: 0.2102, T_mean: 0.9890
2025-04-01 13:57:43,286 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0687451..1.917734].
2025-04-01 13:57:43,288 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0687451..108.55557].
2025-04-01 13:57:43,292 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0686605..153.25804].
2025-04-01 13:57:43,435 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0686219..1.9031312].
2025-04-01 13:57:43,436 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0685372..108.54971].
2025-04-01 13:57:43,439 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0685372..153.26964].
2025-04-01 13:57:48,589 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:48,589 - utils.training - INFO - Global: J&F: 0.1592, J_mean: 0.2874, T_mean: 0.9935
2025-04-01 13:57:53,989 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:53,989 - utils.training - INFO - Global: J&F: 0.0916, J_mean: 0.1466, T_mean: 0.9936
2025-04-01 13:57:56,880 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.949214..2.3596659].
2025-04-01 13:57:56,882 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.949214..108.94081].
2025-04-01 13:57:56,884 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.949214..140.1131].
2025-04-01 13:57:57,016 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9497089..2.3601756].
2025-04-01 13:57:57,018 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9497089..108.93564].
2025-04-01 13:57:57,021 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9497089..140.07967].
2025-04-01 13:57:59,593 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:57:59,593 - utils.training - INFO - Global: J&F: 0.0957, J_mean: 0.1527, T_mean: 0.9912
2025-04-01 13:58:05,144 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:58:05,145 - utils.training - INFO - Global: J&F: 0.1355, J_mean: 0.2281, T_mean: 0.9870
2025-04-01 13:58:10,697 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:58:10,697 - utils.training - INFO - Global: J&F: 0.1601, J_mean: 0.2535, T_mean: 0.9870
2025-04-01 13:58:11,014 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0763974..2.027336].
2025-04-01 13:58:11,016 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0763974..108.810936].
2025-04-01 13:58:11,019 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0763974..153.52676].
2025-04-01 13:58:11,209 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0764034..2.0239816].
2025-04-01 13:58:11,210 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0764034..108.80959].
2025-04-01 13:58:11,213 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0764034..153.60588].
2025-04-01 13:58:16,491 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:58:16,492 - utils.training - INFO - Global: J&F: 0.1657, J_mean: 0.2795, T_mean: 0.9917
2025-04-01 13:58:21,994 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:58:21,995 - utils.training - INFO - Global: J&F: 0.1367, J_mean: 0.2208, T_mean: 0.9854
2025-04-01 13:58:24,434 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 13:58:24,434 - utils.training - INFO - Global: J&F: 0.1370, J_mean: 0.2266, T_mean: 0.9836
2025-04-01 13:58:24,511 - utils.training - INFO - Epoch 5 validation: val_loss: 1.3578 J&F: 0.1370 J_mean: 0.2266 F_mean: 0.0475 T_mean: 0.9836 instance_stability: 1.0000
2025-04-01 13:58:24,681 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 14:01:24,850 - utils.training - INFO - Epoch 6 completed: Loss: 1.4886, CE: 0.6956, Dice: 0.7930, LR: 0.000042
2025-04-01 14:04:24,298 - utils.training - INFO - Epoch 7 completed: Loss: 1.4856, CE: 0.6953, Dice: 0.7903, LR: 0.000040
2025-04-01 14:04:24,911 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0254374..1.626661].
2025-04-01 14:04:24,917 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0254374..108.65036].
2025-04-01 14:04:24,924 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0254374..139.794].
2025-04-01 14:04:25,131 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0252924..1.6268085].
2025-04-01 14:04:25,135 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0252924..108.65073].
2025-04-01 14:04:25,139 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0252924..139.7996].
2025-04-01 14:04:30,347 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:04:30,347 - utils.training - INFO - Global: J&F: 0.1797, J_mean: 0.2999, T_mean: 0.9903
2025-04-01 14:04:35,850 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:04:35,850 - utils.training - INFO - Global: J&F: 0.1067, J_mean: 0.1838, T_mean: 0.9936
2025-04-01 14:04:38,754 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.01049..2.2236395].
2025-04-01 14:04:38,755 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.01049..108.88946].
2025-04-01 14:04:38,758 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.01049..108.87892].
2025-04-01 14:04:38,921 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0101502..2.2127419].
2025-04-01 14:04:38,923 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0101502..108.88509].
2025-04-01 14:04:38,926 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0101502..108.87774].
2025-04-01 14:04:41,560 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:04:41,560 - utils.training - INFO - Global: J&F: 0.1343, J_mean: 0.2329, T_mean: 0.9961
2025-04-01 14:04:46,961 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:04:46,961 - utils.training - INFO - Global: J&F: 0.1129, J_mean: 0.1866, T_mean: 0.9912
2025-04-01 14:04:52,377 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:04:52,378 - utils.training - INFO - Global: J&F: 0.1521, J_mean: 0.2569, T_mean: 0.9889
2025-04-01 14:04:52,696 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399395].
2025-04-01 14:04:52,699 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.00223].
2025-04-01 14:04:52,703 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.71114].
2025-04-01 14:04:52,834 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399395].
2025-04-01 14:04:52,836 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.9788].
2025-04-01 14:04:52,839 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.72803].
2025-04-01 14:04:58,019 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:04:58,020 - utils.training - INFO - Global: J&F: 0.1173, J_mean: 0.2018, T_mean: 0.9960
2025-04-01 14:05:03,427 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:03,428 - utils.training - INFO - Global: J&F: 0.0949, J_mean: 0.1437, T_mean: 0.9933
2025-04-01 14:05:06,325 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9914852..2.4530983].
2025-04-01 14:05:06,328 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9914852..108.98124].
2025-04-01 14:05:06,331 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9914852..140.17252].
2025-04-01 14:05:06,473 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9915781..2.4530034].
2025-04-01 14:05:06,474 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9915781..108.9812].
2025-04-01 14:05:06,477 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9915781..140.1812].
2025-04-01 14:05:09,064 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:09,064 - utils.training - INFO - Global: J&F: 0.1084, J_mean: 0.1730, T_mean: 0.9921
2025-04-01 14:05:14,592 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:14,593 - utils.training - INFO - Global: J&F: 0.1121, J_mean: 0.1870, T_mean: 0.9893
2025-04-01 14:05:20,052 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:20,053 - utils.training - INFO - Global: J&F: 0.1491, J_mean: 0.2466, T_mean: 0.9870
2025-04-01 14:05:20,369 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.2900078].
2025-04-01 14:05:20,371 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.916].
2025-04-01 14:05:20,374 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.61671].
2025-04-01 14:05:20,565 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.2926543].
2025-04-01 14:05:20,567 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.91706].
2025-04-01 14:05:20,570 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.70453].
2025-04-01 14:05:25,908 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:25,908 - utils.training - INFO - Global: J&F: 0.1648, J_mean: 0.2797, T_mean: 0.9934
2025-04-01 14:05:31,398 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:31,399 - utils.training - INFO - Global: J&F: 0.1569, J_mean: 0.2615, T_mean: 0.9880
2025-04-01 14:05:33,780 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:05:33,781 - utils.training - INFO - Global: J&F: 0.1593, J_mean: 0.2617, T_mean: 0.9847
2025-04-01 14:05:33,862 - utils.training - INFO - Epoch 7 validation: val_loss: 1.3733 J&F: 0.1593 J_mean: 0.2617 F_mean: 0.0570 T_mean: 0.9847 instance_stability: 1.0000
2025-04-01 14:08:33,280 - utils.training - INFO - Epoch 8 completed: Loss: 1.4821, CE: 0.6949, Dice: 0.7871, LR: 0.000037
2025-04-01 14:11:33,208 - utils.training - INFO - Epoch 9 completed: Loss: 1.4795, CE: 0.6934, Dice: 0.7861, LR: 0.000035
2025-04-01 14:11:33,773 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0856214..1.9761236].
2025-04-01 14:11:33,778 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0856214..108.79045].
2025-04-01 14:11:33,784 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0856214..139.92165].
2025-04-01 14:11:34,007 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0855684..1.9761776].
2025-04-01 14:11:34,009 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0855684..108.790474].
2025-04-01 14:11:34,012 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0855684..139.92805].
2025-04-01 14:11:39,257 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:11:39,257 - utils.training - INFO - Global: J&F: 0.1662, J_mean: 0.2757, T_mean: 0.9866
2025-04-01 14:11:44,679 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:11:44,679 - utils.training - INFO - Global: J&F: 0.1175, J_mean: 0.2010, T_mean: 0.9892
2025-04-01 14:11:47,604 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.1035457].
2025-04-01 14:11:47,606 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.841415].
2025-04-01 14:11:47,609 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0965292..108.82334].
2025-04-01 14:11:47,774 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.093125..2.0957456].
2025-04-01 14:11:47,776 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.093125..108.833534].
2025-04-01 14:11:47,780 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.093125..108.82109].
2025-04-01 14:11:50,430 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:11:50,430 - utils.training - INFO - Global: J&F: 0.1810, J_mean: 0.3135, T_mean: 0.9935
2025-04-01 14:11:55,869 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:11:55,869 - utils.training - INFO - Global: J&F: 0.1306, J_mean: 0.2134, T_mean: 0.9879
2025-04-01 14:12:01,351 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:01,352 - utils.training - INFO - Global: J&F: 0.1398, J_mean: 0.2246, T_mean: 0.9888
2025-04-01 14:12:01,671 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.9995157].
2025-04-01 14:12:01,673 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.42214].
2025-04-01 14:12:01,676 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.27213].
2025-04-01 14:12:01,827 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.9638199].
2025-04-01 14:12:01,828 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.43766].
2025-04-01 14:12:01,832 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.28468].
2025-04-01 14:12:07,020 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:07,020 - utils.training - INFO - Global: J&F: 0.1268, J_mean: 0.2182, T_mean: 0.9947
2025-04-01 14:12:12,454 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:12,454 - utils.training - INFO - Global: J&F: 0.1074, J_mean: 0.1646, T_mean: 0.9935
2025-04-01 14:12:15,380 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9336301..2.3949935].
2025-04-01 14:12:15,383 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9336301..108.958].
2025-04-01 14:12:15,386 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9336301..140.15154].
2025-04-01 14:12:15,529 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9341947..2.3944192].
2025-04-01 14:12:15,531 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9341947..108.95777].
2025-04-01 14:12:15,534 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9341947..140.15776].
2025-04-01 14:12:18,153 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:18,153 - utils.training - INFO - Global: J&F: 0.1087, J_mean: 0.1748, T_mean: 0.9876
2025-04-01 14:12:23,702 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:23,702 - utils.training - INFO - Global: J&F: 0.1289, J_mean: 0.2150, T_mean: 0.9838
2025-04-01 14:12:29,205 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:29,205 - utils.training - INFO - Global: J&F: 0.1601, J_mean: 0.2553, T_mean: 0.9854
2025-04-01 14:12:29,525 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.071336].
2025-04-01 14:12:29,528 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.82854].
2025-04-01 14:12:29,531 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.53375].
2025-04-01 14:12:29,697 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.0708349].
2025-04-01 14:12:29,699 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.82833].
2025-04-01 14:12:29,702 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.6204].
2025-04-01 14:12:34,986 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:34,986 - utils.training - INFO - Global: J&F: 0.1826, J_mean: 0.3015, T_mean: 0.9890
2025-04-01 14:12:40,464 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:40,464 - utils.training - INFO - Global: J&F: 0.1617, J_mean: 0.2655, T_mean: 0.9817
2025-04-01 14:12:42,936 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:12:42,936 - utils.training - INFO - Global: J&F: 0.1197, J_mean: 0.1859, T_mean: 0.9789
2025-04-01 14:12:43,012 - utils.training - INFO - Epoch 9 validation: val_loss: 1.2585 J&F: 0.1197 J_mean: 0.1859 F_mean: 0.0534 T_mean: 0.9789 instance_stability: 1.0000
2025-04-01 14:12:43,180 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 14:12:43,327 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 14:15:43,357 - utils.training - INFO - Epoch 10 completed: Loss: 1.4771, CE: 0.6926, Dice: 0.7845, LR: 0.000032
2025-04-01 14:18:45,109 - utils.training - INFO - Epoch 11 completed: Loss: 1.4749, CE: 0.6916, Dice: 0.7833, LR: 0.000029
2025-04-01 14:18:45,686 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399746].
2025-04-01 14:18:45,690 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 14:18:45,695 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 14:18:45,894 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399746].
2025-04-01 14:18:45,896 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 14:18:45,899 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 14:18:51,123 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:18:51,124 - utils.training - INFO - Global: J&F: 0.1589, J_mean: 0.2590, T_mean: 0.9861
2025-04-01 14:18:56,572 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:18:56,572 - utils.training - INFO - Global: J&F: 0.1070, J_mean: 0.1791, T_mean: 0.9903
2025-04-01 14:18:59,471 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9563022..2.419691].
2025-04-01 14:18:59,473 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9563022..108.96788].
2025-04-01 14:18:59,476 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9563022..108.96788].
2025-04-01 14:18:59,625 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9557065..2.4202979].
2025-04-01 14:18:59,627 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9557065..108.96812].
2025-04-01 14:18:59,630 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9557065..108.96812].
2025-04-01 14:19:02,291 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:02,291 - utils.training - INFO - Global: J&F: 0.1435, J_mean: 0.2490, T_mean: 0.9939
2025-04-01 14:19:07,728 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:07,728 - utils.training - INFO - Global: J&F: 0.1397, J_mean: 0.2308, T_mean: 0.9889
2025-04-01 14:19:13,145 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:13,145 - utils.training - INFO - Global: J&F: 0.1332, J_mean: 0.2207, T_mean: 0.9836
2025-04-01 14:19:13,464 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.2791498].
2025-04-01 14:19:13,466 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.75359].
2025-04-01 14:19:13,469 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.34811].
2025-04-01 14:19:13,633 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.2630517].
2025-04-01 14:19:13,635 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.7398].
2025-04-01 14:19:13,639 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.361].
2025-04-01 14:19:18,801 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:18,801 - utils.training - INFO - Global: J&F: 0.1039, J_mean: 0.1836, T_mean: 0.9927
2025-04-01 14:19:24,214 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:24,214 - utils.training - INFO - Global: J&F: 0.0992, J_mean: 0.1562, T_mean: 0.9924
2025-04-01 14:19:27,124 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.482774].
2025-04-01 14:19:27,127 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.988304].
2025-04-01 14:19:27,131 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.16844].
2025-04-01 14:19:27,272 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4992144].
2025-04-01 14:19:27,274 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.989494].
2025-04-01 14:19:27,277 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.10474].
2025-04-01 14:19:29,868 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:29,869 - utils.training - INFO - Global: J&F: 0.0989, J_mean: 0.1574, T_mean: 0.9901
2025-04-01 14:19:35,405 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:35,405 - utils.training - INFO - Global: J&F: 0.1155, J_mean: 0.1922, T_mean: 0.9834
2025-04-01 14:19:40,901 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:40,901 - utils.training - INFO - Global: J&F: 0.1363, J_mean: 0.2250, T_mean: 0.9869
2025-04-01 14:19:41,230 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.0032635].
2025-04-01 14:19:41,232 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.80131].
2025-04-01 14:19:41,235 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.51833].
2025-04-01 14:19:41,424 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.0021307].
2025-04-01 14:19:41,425 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.80085].
2025-04-01 14:19:41,429 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.60002].
2025-04-01 14:19:46,704 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:46,705 - utils.training - INFO - Global: J&F: 0.1616, J_mean: 0.2709, T_mean: 0.9899
2025-04-01 14:19:52,108 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:52,108 - utils.training - INFO - Global: J&F: 0.1436, J_mean: 0.2349, T_mean: 0.9847
2025-04-01 14:19:54,551 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:19:54,551 - utils.training - INFO - Global: J&F: 0.1369, J_mean: 0.2194, T_mean: 0.9813
2025-04-01 14:19:54,630 - utils.training - INFO - Epoch 11 validation: val_loss: 1.2709 J&F: 0.1369 J_mean: 0.2194 F_mean: 0.0544 T_mean: 0.9813 instance_stability: 1.0000
2025-04-01 14:22:56,046 - utils.training - INFO - Epoch 12 completed: Loss: 1.4736, CE: 0.6911, Dice: 0.7825, LR: 0.000026
2025-04-01 14:25:57,802 - utils.training - INFO - Epoch 13 completed: Loss: 1.4712, CE: 0.6902, Dice: 0.7810, LR: 0.000022
2025-04-01 14:25:58,401 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9067367..2.3226633].
2025-04-01 14:25:58,405 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9067367..108.92361].
2025-04-01 14:25:58,417 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9067367..140.12361].
2025-04-01 14:25:58,615 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9063786..2.3187819].
2025-04-01 14:25:58,617 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9063786..108.91892].
2025-04-01 14:25:58,620 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9063786..140.1217].
2025-04-01 14:26:03,854 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:03,854 - utils.training - INFO - Global: J&F: 0.1599, J_mean: 0.2592, T_mean: 0.9863
2025-04-01 14:26:09,376 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:09,376 - utils.training - INFO - Global: J&F: 0.1513, J_mean: 0.2623, T_mean: 0.9892
2025-04-01 14:26:12,312 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9629356..2.459044].
2025-04-01 14:26:12,314 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9629356..108.98362].
2025-04-01 14:26:12,318 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9629356..108.98362].
2025-04-01 14:26:12,512 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9623579..2.4596314].
2025-04-01 14:26:12,514 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9623579..108.98386].
2025-04-01 14:26:12,517 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9623579..108.98386].
2025-04-01 14:26:15,200 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:15,200 - utils.training - INFO - Global: J&F: 0.1441, J_mean: 0.2469, T_mean: 0.9947
2025-04-01 14:26:20,668 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:20,668 - utils.training - INFO - Global: J&F: 0.1478, J_mean: 0.2474, T_mean: 0.9888
2025-04-01 14:26:26,173 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:26,174 - utils.training - INFO - Global: J&F: 0.1286, J_mean: 0.2164, T_mean: 0.9852
2025-04-01 14:26:26,522 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.8825911].
2025-04-01 14:26:26,524 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.39203].
2025-04-01 14:26:26,528 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.23663].
2025-04-01 14:26:26,675 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.8676023].
2025-04-01 14:26:26,677 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.34012].
2025-04-01 14:26:26,679 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.24832].
2025-04-01 14:26:31,822 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:31,822 - utils.training - INFO - Global: J&F: 0.1580, J_mean: 0.2846, T_mean: 0.9942
2025-04-01 14:26:37,271 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:37,272 - utils.training - INFO - Global: J&F: 0.1198, J_mean: 0.1823, T_mean: 0.9928
2025-04-01 14:26:40,189 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.107168..1.9222534].
2025-04-01 14:26:40,191 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.107168..108.760185].
2025-04-01 14:26:40,194 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.107168..139.9457].
2025-04-01 14:26:40,339 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.107176..1.9389716].
2025-04-01 14:26:40,341 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.107176..108.76904].
2025-04-01 14:26:40,344 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.107176..139.89464].
2025-04-01 14:26:42,980 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:42,980 - utils.training - INFO - Global: J&F: 0.1002, J_mean: 0.1588, T_mean: 0.9878
2025-04-01 14:26:48,476 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:48,476 - utils.training - INFO - Global: J&F: 0.1291, J_mean: 0.2118, T_mean: 0.9812
2025-04-01 14:26:54,051 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:54,052 - utils.training - INFO - Global: J&F: 0.1297, J_mean: 0.2061, T_mean: 0.9860
2025-04-01 14:26:54,373 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:26:54,375 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:26:54,379 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89957].
2025-04-01 14:26:54,542 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:26:54,543 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:26:54,547 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89957].
2025-04-01 14:26:59,807 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:26:59,807 - utils.training - INFO - Global: J&F: 0.1852, J_mean: 0.3058, T_mean: 0.9892
2025-04-01 14:27:05,277 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:27:05,277 - utils.training - INFO - Global: J&F: 0.1408, J_mean: 0.2357, T_mean: 0.9797
2025-04-01 14:27:07,694 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:27:07,694 - utils.training - INFO - Global: J&F: 0.1261, J_mean: 0.2038, T_mean: 0.9786
2025-04-01 14:27:07,774 - utils.training - INFO - Epoch 13 validation: val_loss: 1.2619 J&F: 0.1261 J_mean: 0.2038 F_mean: 0.0485 T_mean: 0.9786 instance_stability: 1.0000
2025-04-01 14:30:09,712 - utils.training - INFO - Epoch 14 completed: Loss: 1.4701, CE: 0.6898, Dice: 0.7804, LR: 0.000019
2025-04-01 14:33:11,480 - utils.training - INFO - Epoch 15 completed: Loss: 1.4683, CE: 0.6891, Dice: 0.7792, LR: 0.000016
2025-04-01 14:33:12,070 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0695431..2.3709006].
2025-04-01 14:33:12,074 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0695431..108.94836].
2025-04-01 14:33:12,083 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0695431..140.0941].
2025-04-01 14:33:12,318 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.06946..2.373046].
2025-04-01 14:33:12,320 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.06946..108.94829].
2025-04-01 14:33:12,324 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.06946..140.1004].
2025-04-01 14:33:17,558 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:17,558 - utils.training - INFO - Global: J&F: 0.1841, J_mean: 0.3064, T_mean: 0.9837
2025-04-01 14:33:23,056 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:23,056 - utils.training - INFO - Global: J&F: 0.1506, J_mean: 0.2633, T_mean: 0.9889
2025-04-01 14:33:25,962 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3755794].
2025-04-01 14:33:25,964 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92043].
2025-04-01 14:33:25,966 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.927734].
2025-04-01 14:33:26,117 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.367185].
2025-04-01 14:33:26,119 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.9274].
2025-04-01 14:33:26,122 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92485].
2025-04-01 14:33:28,816 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:28,816 - utils.training - INFO - Global: J&F: 0.1738, J_mean: 0.3134, T_mean: 0.9924
2025-04-01 14:33:34,266 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:34,267 - utils.training - INFO - Global: J&F: 0.1733, J_mean: 0.2963, T_mean: 0.9880
2025-04-01 14:33:39,703 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:39,703 - utils.training - INFO - Global: J&F: 0.1548, J_mean: 0.2582, T_mean: 0.9815
2025-04-01 14:33:40,022 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399257].
2025-04-01 14:33:40,025 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 14:33:40,029 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.72565].
2025-04-01 14:33:40,162 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399257].
2025-04-01 14:33:40,164 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 14:33:40,168 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.74335].
2025-04-01 14:33:45,314 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:45,314 - utils.training - INFO - Global: J&F: 0.1144, J_mean: 0.2039, T_mean: 0.9917
2025-04-01 14:33:50,739 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:50,739 - utils.training - INFO - Global: J&F: 0.0755, J_mean: 0.1182, T_mean: 0.9900
2025-04-01 14:33:53,649 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:33:53,652 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:33:53,655 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.20335].
2025-04-01 14:33:53,797 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:33:53,798 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:33:53,802 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 14:33:56,422 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:33:56,423 - utils.training - INFO - Global: J&F: 0.0828, J_mean: 0.1366, T_mean: 0.9871
2025-04-01 14:34:01,960 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:34:01,960 - utils.training - INFO - Global: J&F: 0.1337, J_mean: 0.2236, T_mean: 0.9794
2025-04-01 14:34:07,447 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:34:07,448 - utils.training - INFO - Global: J&F: 0.1416, J_mean: 0.2323, T_mean: 0.9816
2025-04-01 14:34:07,790 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7487622..1.8433093].
2025-04-01 14:34:07,792 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7487622..108.73732].
2025-04-01 14:34:07,795 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7487622..153.4736].
2025-04-01 14:34:07,983 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7488604..1.8413985].
2025-04-01 14:34:07,984 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7488604..108.73656].
2025-04-01 14:34:07,987 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7488604..153.54274].
2025-04-01 14:34:13,308 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:34:13,308 - utils.training - INFO - Global: J&F: 0.1377, J_mean: 0.2306, T_mean: 0.9864
2025-04-01 14:34:18,792 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:34:18,792 - utils.training - INFO - Global: J&F: 0.1453, J_mean: 0.2475, T_mean: 0.9775
2025-04-01 14:34:21,255 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:34:21,256 - utils.training - INFO - Global: J&F: 0.1161, J_mean: 0.1823, T_mean: 0.9780
2025-04-01 14:34:21,338 - utils.training - INFO - Epoch 15 validation: val_loss: 1.2322 J&F: 0.1161 J_mean: 0.1823 F_mean: 0.0499 T_mean: 0.9780 instance_stability: 1.0000
2025-04-01 14:34:21,509 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 14:37:23,319 - utils.training - INFO - Epoch 16 completed: Loss: 1.4679, CE: 0.6888, Dice: 0.7791, LR: 0.000014
2025-04-01 14:40:24,690 - utils.training - INFO - Epoch 17 completed: Loss: 1.4659, CE: 0.6880, Dice: 0.7780, LR: 0.000011
2025-04-01 14:40:25,237 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3683336].
2025-04-01 14:40:25,241 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.94503].
2025-04-01 14:40:25,248 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.08607].
2025-04-01 14:40:25,462 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3689716].
2025-04-01 14:40:25,464 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.947586].
2025-04-01 14:40:25,467 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.09259].
2025-04-01 14:40:30,657 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:40:30,658 - utils.training - INFO - Global: J&F: 0.1800, J_mean: 0.2971, T_mean: 0.9897
2025-04-01 14:40:36,119 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:40:36,119 - utils.training - INFO - Global: J&F: 0.1488, J_mean: 0.2584, T_mean: 0.9915
2025-04-01 14:40:39,039 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6055534].
2025-04-01 14:40:39,041 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.04222].
2025-04-01 14:40:39,044 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.106982..109.02483].
2025-04-01 14:40:39,201 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.1036067..2.6209562].
2025-04-01 14:40:39,204 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.1036067..109.03981].
2025-04-01 14:40:39,207 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.1036067..109.011856].
2025-04-01 14:40:41,864 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:40:41,865 - utils.training - INFO - Global: J&F: 0.1554, J_mean: 0.2713, T_mean: 0.9956
2025-04-01 14:40:47,296 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:40:47,296 - utils.training - INFO - Global: J&F: 0.1336, J_mean: 0.2294, T_mean: 0.9903
2025-04-01 14:40:52,751 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:40:52,752 - utils.training - INFO - Global: J&F: 0.1518, J_mean: 0.2437, T_mean: 0.9849
2025-04-01 14:40:53,073 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9415895..2.2472737].
2025-04-01 14:40:53,075 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9415895..108.71281].
2025-04-01 14:40:53,078 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9413989..153.36835].
2025-04-01 14:40:53,228 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9411455..2.2321467].
2025-04-01 14:40:53,231 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9411455..108.72561].
2025-04-01 14:40:53,234 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9409548..153.38072].
2025-04-01 14:40:58,375 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:40:58,375 - utils.training - INFO - Global: J&F: 0.1497, J_mean: 0.2676, T_mean: 0.9955
2025-04-01 14:41:03,815 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:03,816 - utils.training - INFO - Global: J&F: 0.1148, J_mean: 0.1726, T_mean: 0.9936
2025-04-01 14:41:06,755 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0556045..2.555969].
2025-04-01 14:41:06,757 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0556045..109.022385].
2025-04-01 14:41:06,761 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0556045..140.14368].
2025-04-01 14:41:06,908 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0557022..2.5583115].
2025-04-01 14:41:06,910 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0557022..109.02332].
2025-04-01 14:41:06,912 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0557022..140.15703].
2025-04-01 14:41:09,563 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:09,564 - utils.training - INFO - Global: J&F: 0.1083, J_mean: 0.1765, T_mean: 0.9906
2025-04-01 14:41:15,049 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:15,049 - utils.training - INFO - Global: J&F: 0.1142, J_mean: 0.1930, T_mean: 0.9838
2025-04-01 14:41:20,614 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:20,614 - utils.training - INFO - Global: J&F: 0.1522, J_mean: 0.2416, T_mean: 0.9859
2025-04-01 14:41:20,950 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5941621..2.2899735].
2025-04-01 14:41:20,952 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5941621..108.91599].
2025-04-01 14:41:20,956 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5941621..153.76035].
2025-04-01 14:41:21,133 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5944241..2.2897065].
2025-04-01 14:41:21,135 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5944241..108.915886].
2025-04-01 14:41:21,138 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5944241..153.76189].
2025-04-01 14:41:26,439 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:26,439 - utils.training - INFO - Global: J&F: 0.1690, J_mean: 0.2801, T_mean: 0.9915
2025-04-01 14:41:31,898 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:31,898 - utils.training - INFO - Global: J&F: 0.1254, J_mean: 0.2059, T_mean: 0.9838
2025-04-01 14:41:34,344 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:41:34,345 - utils.training - INFO - Global: J&F: 0.1277, J_mean: 0.2002, T_mean: 0.9821
2025-04-01 14:41:34,426 - utils.training - INFO - Epoch 17 validation: val_loss: 1.2551 J&F: 0.1277 J_mean: 0.2002 F_mean: 0.0553 T_mean: 0.9821 instance_stability: 1.0000
2025-04-01 14:44:36,328 - utils.training - INFO - Epoch 18 completed: Loss: 1.4654, CE: 0.6879, Dice: 0.7775, LR: 0.000009
2025-04-01 14:47:38,161 - utils.training - INFO - Epoch 19 completed: Loss: 1.4654, CE: 0.6881, Dice: 0.7773, LR: 0.000007
2025-04-01 14:47:38,752 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.931108..2.1667318].
2025-04-01 14:47:38,756 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.931108..108.86669].
2025-04-01 14:47:38,766 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.931108..140.0226].
2025-04-01 14:47:38,947 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9308221..2.1718078].
2025-04-01 14:47:38,949 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9308221..108.86657].
2025-04-01 14:47:38,952 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9308221..140.02815].
2025-04-01 14:47:44,179 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:47:44,179 - utils.training - INFO - Global: J&F: 0.1734, J_mean: 0.2883, T_mean: 0.9856
2025-04-01 14:47:49,652 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:47:49,652 - utils.training - INFO - Global: J&F: 0.1488, J_mean: 0.2570, T_mean: 0.9891
2025-04-01 14:47:52,586 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:47:52,588 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:47:52,591 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:47:52,763 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:47:52,765 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:47:52,768 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:47:55,422 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:47:55,422 - utils.training - INFO - Global: J&F: 0.1615, J_mean: 0.2834, T_mean: 0.9926
2025-04-01 14:48:00,842 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:00,842 - utils.training - INFO - Global: J&F: 0.1711, J_mean: 0.2953, T_mean: 0.9895
2025-04-01 14:48:06,292 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:06,292 - utils.training - INFO - Global: J&F: 0.1384, J_mean: 0.2273, T_mean: 0.9797
2025-04-01 14:48:06,612 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9334857..2.1213105].
2025-04-01 14:48:06,613 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9334857..108.525314].
2025-04-01 14:48:06,617 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9334857..153.34914].
2025-04-01 14:48:06,781 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9329559..2.1075034].
2025-04-01 14:48:06,783 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9329559..108.55633].
2025-04-01 14:48:06,786 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9329559..153.36122].
2025-04-01 14:48:11,993 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:11,993 - utils.training - INFO - Global: J&F: 0.1194, J_mean: 0.2110, T_mean: 0.9903
2025-04-01 14:48:17,385 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:17,385 - utils.training - INFO - Global: J&F: 0.0830, J_mean: 0.1264, T_mean: 0.9898
2025-04-01 14:48:20,289 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9614959..2.3832142].
2025-04-01 14:48:20,291 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9614959..108.953285].
2025-04-01 14:48:20,295 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9614959..140.14084].
2025-04-01 14:48:20,438 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9618957..2.3828073].
2025-04-01 14:48:20,440 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9618957..108.95249].
2025-04-01 14:48:20,443 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9618957..140.13693].
2025-04-01 14:48:23,096 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:23,097 - utils.training - INFO - Global: J&F: 0.0879, J_mean: 0.1471, T_mean: 0.9849
2025-04-01 14:48:28,612 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:28,612 - utils.training - INFO - Global: J&F: 0.1226, J_mean: 0.2084, T_mean: 0.9802
2025-04-01 14:48:34,130 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:34,130 - utils.training - INFO - Global: J&F: 0.1351, J_mean: 0.2188, T_mean: 0.9844
2025-04-01 14:48:34,445 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.035849..2.588019].
2025-04-01 14:48:34,447 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.035849..109.03521].
2025-04-01 14:48:34,451 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.035849..153.80037].
2025-04-01 14:48:34,639 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0358706..2.5879967].
2025-04-01 14:48:34,640 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0358706..109.0352].
2025-04-01 14:48:34,644 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0358706..153.8748].
2025-04-01 14:48:39,956 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:39,956 - utils.training - INFO - Global: J&F: 0.1541, J_mean: 0.2545, T_mean: 0.9870
2025-04-01 14:48:45,382 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:45,383 - utils.training - INFO - Global: J&F: 0.1221, J_mean: 0.2049, T_mean: 0.9803
2025-04-01 14:48:47,826 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:48:47,827 - utils.training - INFO - Global: J&F: 0.1294, J_mean: 0.2056, T_mean: 0.9770
2025-04-01 14:48:47,907 - utils.training - INFO - Epoch 19 validation: val_loss: 1.2137 J&F: 0.1294 J_mean: 0.2056 F_mean: 0.0533 T_mean: 0.9770 instance_stability: 1.0000
2025-04-01 14:48:48,078 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 14:48:48,249 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 14:51:49,876 - utils.training - INFO - Epoch 20 completed: Loss: 1.4637, CE: 0.6876, Dice: 0.7762, LR: 0.000005
2025-04-01 14:54:51,148 - utils.training - INFO - Epoch 21 completed: Loss: 1.4627, CE: 0.6868, Dice: 0.7759, LR: 0.000003
2025-04-01 14:54:51,760 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9797579..2.0246959].
2025-04-01 14:54:51,762 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9797579..108.809616].
2025-04-01 14:54:51,766 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9797579..139.95506].
2025-04-01 14:54:51,984 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9795299..2.024928].
2025-04-01 14:54:51,986 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9795299..108.80979].
2025-04-01 14:54:51,990 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9795299..139.96103].
2025-04-01 14:54:57,225 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:54:57,225 - utils.training - INFO - Global: J&F: 0.2075, J_mean: 0.3459, T_mean: 0.9881
2025-04-01 14:55:02,680 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:02,680 - utils.training - INFO - Global: J&F: 0.1498, J_mean: 0.2615, T_mean: 0.9903
2025-04-01 14:55:05,611 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:55:05,613 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:55:05,616 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:55:05,767 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 14:55:05,768 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:55:05,771 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 14:55:08,447 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:08,447 - utils.training - INFO - Global: J&F: 0.1824, J_mean: 0.3164, T_mean: 0.9945
2025-04-01 14:55:13,888 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:13,888 - utils.training - INFO - Global: J&F: 0.1477, J_mean: 0.2572, T_mean: 0.9896
2025-04-01 14:55:19,314 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:19,314 - utils.training - INFO - Global: J&F: 0.1562, J_mean: 0.2589, T_mean: 0.9835
2025-04-01 14:55:19,634 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4956906].
2025-04-01 14:55:19,636 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.64023].
2025-04-01 14:55:19,639 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.40393].
2025-04-01 14:55:19,798 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4783807].
2025-04-01 14:55:19,800 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.6543].
2025-04-01 14:55:19,803 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.41763].
2025-04-01 14:55:24,955 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:24,955 - utils.training - INFO - Global: J&F: 0.1812, J_mean: 0.3270, T_mean: 0.9930
2025-04-01 14:55:30,361 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:30,362 - utils.training - INFO - Global: J&F: 0.1154, J_mean: 0.1760, T_mean: 0.9906
2025-04-01 14:55:33,259 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399846].
2025-04-01 14:55:33,261 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 14:55:33,264 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25423].
2025-04-01 14:55:33,425 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399846].
2025-04-01 14:55:33,426 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05599].
2025-04-01 14:55:33,430 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 14:55:36,047 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:36,048 - utils.training - INFO - Global: J&F: 0.1098, J_mean: 0.1754, T_mean: 0.9877
2025-04-01 14:55:41,588 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:41,588 - utils.training - INFO - Global: J&F: 0.1248, J_mean: 0.2081, T_mean: 0.9829
2025-04-01 14:55:47,128 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:47,129 - utils.training - INFO - Global: J&F: 0.1698, J_mean: 0.2723, T_mean: 0.9872
2025-04-01 14:55:47,463 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0194013..2.2872262].
2025-04-01 14:55:47,465 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8117069..108.914894].
2025-04-01 14:55:47,469 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7892404..153.62076].
2025-04-01 14:55:47,645 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0818102..2.2901657].
2025-04-01 14:55:47,646 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9615917..108.91542].
2025-04-01 14:55:47,650 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8229471..153.71185].
2025-04-01 14:55:53,020 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:53,020 - utils.training - INFO - Global: J&F: 0.1852, J_mean: 0.3046, T_mean: 0.9895
2025-04-01 14:55:58,452 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:55:58,453 - utils.training - INFO - Global: J&F: 0.1472, J_mean: 0.2454, T_mean: 0.9835
2025-04-01 14:56:00,884 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 14:56:00,884 - utils.training - INFO - Global: J&F: 0.1150, J_mean: 0.1841, T_mean: 0.9801
2025-04-01 14:56:00,966 - utils.training - INFO - Epoch 21 validation: val_loss: 1.2366 J&F: 0.1150 J_mean: 0.1841 F_mean: 0.0459 T_mean: 0.9801 instance_stability: 1.0000
2025-04-01 14:59:02,410 - utils.training - INFO - Epoch 22 completed: Loss: 1.4637, CE: 0.6874, Dice: 0.7763, LR: 0.000002
2025-04-01 15:02:04,192 - utils.training - INFO - Epoch 23 completed: Loss: 1.4632, CE: 0.6870, Dice: 0.7762, LR: 0.000001
2025-04-01 15:02:04,743 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:02:04,746 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:02:04,751 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 15:02:04,976 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:02:04,978 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:02:04,981 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 15:02:10,205 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:10,205 - utils.training - INFO - Global: J&F: 0.1920, J_mean: 0.3164, T_mean: 0.9873
2025-04-01 15:02:15,675 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:15,676 - utils.training - INFO - Global: J&F: 0.1562, J_mean: 0.2758, T_mean: 0.9900
2025-04-01 15:02:18,570 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8471124..1.4469782].
2025-04-01 15:02:18,572 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8471124..108.57879].
2025-04-01 15:02:18,574 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8283991..108.574646].
2025-04-01 15:02:18,726 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8256593..1.4426559].
2025-04-01 15:02:18,728 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8256593..108.577065].
2025-04-01 15:02:18,731 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8256593..108.57421].
2025-04-01 15:02:21,390 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:21,390 - utils.training - INFO - Global: J&F: 0.1601, J_mean: 0.2794, T_mean: 0.9942
2025-04-01 15:02:26,855 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:26,856 - utils.training - INFO - Global: J&F: 0.1545, J_mean: 0.2670, T_mean: 0.9898
2025-04-01 15:02:32,348 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:32,348 - utils.training - INFO - Global: J&F: 0.1422, J_mean: 0.2301, T_mean: 0.9831
2025-04-01 15:02:32,669 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8456316..2.3405733].
2025-04-01 15:02:32,671 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8456316..108.936226].
2025-04-01 15:02:32,675 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8456316..153.4689].
2025-04-01 15:02:32,835 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8448384..2.3413806].
2025-04-01 15:02:32,837 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8448384..108.93318].
2025-04-01 15:02:32,841 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8448384..153.48204].
2025-04-01 15:02:38,009 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:38,010 - utils.training - INFO - Global: J&F: 0.1061, J_mean: 0.1874, T_mean: 0.9938
2025-04-01 15:02:43,390 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:43,391 - utils.training - INFO - Global: J&F: 0.0921, J_mean: 0.1422, T_mean: 0.9919
2025-04-01 15:02:46,309 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0187764..2.3173575].
2025-04-01 15:02:46,311 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0187764..108.92045].
2025-04-01 15:02:46,314 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0187764..140.10703].
2025-04-01 15:02:46,461 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0190487..2.3279927].
2025-04-01 15:02:46,462 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0190487..108.92234].
2025-04-01 15:02:46,465 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0190487..140.04962].
2025-04-01 15:02:49,128 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:49,128 - utils.training - INFO - Global: J&F: 0.0902, J_mean: 0.1476, T_mean: 0.9866
2025-04-01 15:02:54,655 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:02:54,655 - utils.training - INFO - Global: J&F: 0.1290, J_mean: 0.2180, T_mean: 0.9797
2025-04-01 15:03:00,135 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:03:00,135 - utils.training - INFO - Global: J&F: 0.1476, J_mean: 0.2411, T_mean: 0.9858
2025-04-01 15:03:00,457 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.4669812..2.2052767].
2025-04-01 15:03:00,460 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.3088771..108.88211].
2025-04-01 15:03:00,463 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.1998956..153.60884].
2025-04-01 15:03:00,652 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.5223709..2.2038462].
2025-04-01 15:03:00,654 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.2715423..108.88154].
2025-04-01 15:03:00,658 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.2105591..153.67764].
2025-04-01 15:03:05,973 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:03:05,974 - utils.training - INFO - Global: J&F: 0.1596, J_mean: 0.2603, T_mean: 0.9897
2025-04-01 15:03:11,441 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:03:11,441 - utils.training - INFO - Global: J&F: 0.1520, J_mean: 0.2528, T_mean: 0.9810
2025-04-01 15:03:13,888 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:03:13,888 - utils.training - INFO - Global: J&F: 0.1286, J_mean: 0.1996, T_mean: 0.9807
2025-04-01 15:03:13,968 - utils.training - INFO - Epoch 23 validation: val_loss: 1.2524 J&F: 0.1286 J_mean: 0.1996 F_mean: 0.0575 T_mean: 0.9807 instance_stability: 1.0000
2025-04-01 15:06:15,501 - utils.training - INFO - Epoch 24 completed: Loss: 1.4631, CE: 0.6870, Dice: 0.7761, LR: 0.000001
2025-04-01 15:09:17,057 - utils.training - INFO - Epoch 25 completed: Loss: 1.4625, CE: 0.6866, Dice: 0.7759, LR: 0.000001
2025-04-01 15:09:17,682 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8842537..1.802283].
2025-04-01 15:09:17,687 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8842537..108.72092].
2025-04-01 15:09:17,700 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8842537..139.88422].
2025-04-01 15:09:17,918 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8838986..1.8082409].
2025-04-01 15:09:17,920 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8838986..108.7233].
2025-04-01 15:09:17,923 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8838986..139.88907].
2025-04-01 15:09:23,137 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:23,137 - utils.training - INFO - Global: J&F: 0.1616, J_mean: 0.2660, T_mean: 0.9857
2025-04-01 15:09:28,624 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:28,624 - utils.training - INFO - Global: J&F: 0.1604, J_mean: 0.2824, T_mean: 0.9899
2025-04-01 15:09:31,543 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399634].
2025-04-01 15:09:31,545 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 15:09:31,548 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 15:09:31,698 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399634].
2025-04-01 15:09:31,700 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 15:09:31,703 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 15:09:34,373 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:34,374 - utils.training - INFO - Global: J&F: 0.1746, J_mean: 0.3074, T_mean: 0.9941
2025-04-01 15:09:39,830 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:39,830 - utils.training - INFO - Global: J&F: 0.1754, J_mean: 0.3068, T_mean: 0.9894
2025-04-01 15:09:45,318 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:45,319 - utils.training - INFO - Global: J&F: 0.1302, J_mean: 0.2162, T_mean: 0.9799
2025-04-01 15:09:45,650 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:09:45,652 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.96179].
2025-04-01 15:09:45,655 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.52782].
2025-04-01 15:09:45,794 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:09:45,796 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.96167].
2025-04-01 15:09:45,799 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.5429].
2025-04-01 15:09:50,962 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:50,962 - utils.training - INFO - Global: J&F: 0.1284, J_mean: 0.2308, T_mean: 0.9895
2025-04-01 15:09:56,388 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:09:56,388 - utils.training - INFO - Global: J&F: 0.0971, J_mean: 0.1514, T_mean: 0.9914
2025-04-01 15:09:59,293 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3494213].
2025-04-01 15:09:59,295 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.93654].
2025-04-01 15:09:59,298 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.12233].
2025-04-01 15:09:59,452 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.355823].
2025-04-01 15:09:59,453 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.9293].
2025-04-01 15:09:59,456 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.0619].
2025-04-01 15:10:02,056 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:10:02,056 - utils.training - INFO - Global: J&F: 0.0942, J_mean: 0.1549, T_mean: 0.9853
2025-04-01 15:10:07,644 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:10:07,645 - utils.training - INFO - Global: J&F: 0.1228, J_mean: 0.2091, T_mean: 0.9794
2025-04-01 15:10:13,188 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:10:13,188 - utils.training - INFO - Global: J&F: 0.1392, J_mean: 0.2262, T_mean: 0.9831
2025-04-01 15:10:13,522 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399312].
2025-04-01 15:10:13,524 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 15:10:13,529 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89804].
2025-04-01 15:10:13,709 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399312].
2025-04-01 15:10:13,711 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05597].
2025-04-01 15:10:13,714 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89954].
2025-04-01 15:10:18,980 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:10:18,981 - utils.training - INFO - Global: J&F: 0.1390, J_mean: 0.2287, T_mean: 0.9891
2025-04-01 15:10:24,470 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:10:24,471 - utils.training - INFO - Global: J&F: 0.1502, J_mean: 0.2535, T_mean: 0.9782
2025-04-01 15:10:26,928 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:10:26,929 - utils.training - INFO - Global: J&F: 0.1206, J_mean: 0.1926, T_mean: 0.9820
2025-04-01 15:10:27,009 - utils.training - INFO - Epoch 25 validation: val_loss: 1.2339 J&F: 0.1206 J_mean: 0.1926 F_mean: 0.0486 T_mean: 0.9820 instance_stability: 1.0000
2025-04-01 15:13:28,829 - utils.training - INFO - Epoch 26 completed: Loss: 1.4627, CE: 0.6870, Dice: 0.7757, LR: 0.000001
2025-04-01 15:16:30,564 - utils.training - INFO - Epoch 27 completed: Loss: 1.4623, CE: 0.6866, Dice: 0.7757, LR: 0.000002
2025-04-01 15:16:31,149 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9449793..2.3219507].
2025-04-01 15:16:31,154 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9449793..108.92878].
2025-04-01 15:16:31,159 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9449793..140.12878].
2025-04-01 15:16:31,386 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9446969..2.3222377].
2025-04-01 15:16:31,387 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9446969..108.928894].
2025-04-01 15:16:31,391 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9446969..140.12889].
2025-04-01 15:16:36,590 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:16:36,590 - utils.training - INFO - Global: J&F: 0.1825, J_mean: 0.3077, T_mean: 0.9860
2025-04-01 15:16:42,015 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:16:42,015 - utils.training - INFO - Global: J&F: 0.1806, J_mean: 0.3122, T_mean: 0.9899
2025-04-01 15:16:44,910 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.639937].
2025-04-01 15:16:44,912 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05598].
2025-04-01 15:16:44,915 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05598].
2025-04-01 15:16:45,071 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.639937].
2025-04-01 15:16:45,073 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05598].
2025-04-01 15:16:45,076 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05598].
2025-04-01 15:16:47,787 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:16:47,788 - utils.training - INFO - Global: J&F: 0.1640, J_mean: 0.2855, T_mean: 0.9943
2025-04-01 15:16:53,226 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:16:53,226 - utils.training - INFO - Global: J&F: 0.1608, J_mean: 0.2762, T_mean: 0.9898
2025-04-01 15:16:58,706 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:16:58,706 - utils.training - INFO - Global: J&F: 0.1332, J_mean: 0.2270, T_mean: 0.9814
2025-04-01 15:16:59,035 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6144495].
2025-04-01 15:16:59,038 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.847595].
2025-04-01 15:16:59,041 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.46547].
2025-04-01 15:16:59,185 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.629968].
2025-04-01 15:16:59,187 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.93029].
2025-04-01 15:16:59,190 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.47946].
2025-04-01 15:17:04,337 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:04,338 - utils.training - INFO - Global: J&F: 0.1453, J_mean: 0.2644, T_mean: 0.9916
2025-04-01 15:17:09,721 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:09,721 - utils.training - INFO - Global: J&F: 0.1066, J_mean: 0.1646, T_mean: 0.9918
2025-04-01 15:17:12,631 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.039926..1.6197895].
2025-04-01 15:17:12,633 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.039926..108.63565].
2025-04-01 15:17:12,637 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.039926..139.82912].
2025-04-01 15:17:12,762 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0399704..1.6317796].
2025-04-01 15:17:12,763 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0399704..108.64405].
2025-04-01 15:17:12,767 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0399704..139.78288].
2025-04-01 15:17:15,450 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:15,450 - utils.training - INFO - Global: J&F: 0.1108, J_mean: 0.1838, T_mean: 0.9846
2025-04-01 15:17:20,983 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:20,984 - utils.training - INFO - Global: J&F: 0.1384, J_mean: 0.2309, T_mean: 0.9810
2025-04-01 15:17:26,550 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:26,550 - utils.training - INFO - Global: J&F: 0.1553, J_mean: 0.2490, T_mean: 0.9866
2025-04-01 15:17:26,866 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.43618].
2025-04-01 15:17:26,868 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.97447].
2025-04-01 15:17:26,871 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.66966].
2025-04-01 15:17:27,037 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.436774].
2025-04-01 15:17:27,039 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.97471].
2025-04-01 15:17:27,042 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.76344].
2025-04-01 15:17:32,393 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:32,393 - utils.training - INFO - Global: J&F: 0.1453, J_mean: 0.2410, T_mean: 0.9899
2025-04-01 15:17:37,907 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:37,907 - utils.training - INFO - Global: J&F: 0.1453, J_mean: 0.2445, T_mean: 0.9817
2025-04-01 15:17:40,365 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:17:40,365 - utils.training - INFO - Global: J&F: 0.1234, J_mean: 0.1975, T_mean: 0.9796
2025-04-01 15:17:40,449 - utils.training - INFO - Epoch 27 validation: val_loss: 1.2082 J&F: 0.1234 J_mean: 0.1975 F_mean: 0.0493 T_mean: 0.9796 instance_stability: 1.0000
2025-04-01 15:17:40,618 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 15:20:42,431 - utils.training - INFO - Epoch 28 completed: Loss: 1.4625, CE: 0.6871, Dice: 0.7754, LR: 0.000003
2025-04-01 15:23:44,160 - utils.training - INFO - Epoch 29 completed: Loss: 1.4634, CE: 0.6872, Dice: 0.7763, LR: 0.000005
2025-04-01 15:23:44,773 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4879234].
2025-04-01 15:23:44,778 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.995094].
2025-04-01 15:23:44,789 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.12915].
2025-04-01 15:23:44,995 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4878385].
2025-04-01 15:23:44,997 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.99513].
2025-04-01 15:23:45,000 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.13608].
2025-04-01 15:23:50,241 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:23:50,242 - utils.training - INFO - Global: J&F: 0.1764, J_mean: 0.2949, T_mean: 0.9867
2025-04-01 15:23:55,683 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:23:55,683 - utils.training - INFO - Global: J&F: 0.1587, J_mean: 0.2737, T_mean: 0.9901
2025-04-01 15:23:58,588 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9579921..2.4000475].
2025-04-01 15:23:58,590 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9579921..108.95954].
2025-04-01 15:23:58,593 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9579921..108.96002].
2025-04-01 15:23:58,747 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.958019..2.4000201].
2025-04-01 15:23:58,749 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.958019..108.959366].
2025-04-01 15:23:58,752 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.958019..108.96001].
2025-04-01 15:24:01,400 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:01,401 - utils.training - INFO - Global: J&F: 0.1609, J_mean: 0.2819, T_mean: 0.9941
2025-04-01 15:24:06,819 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:06,820 - utils.training - INFO - Global: J&F: 0.1705, J_mean: 0.2979, T_mean: 0.9900
2025-04-01 15:24:12,288 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:12,289 - utils.training - INFO - Global: J&F: 0.1355, J_mean: 0.2292, T_mean: 0.9810
2025-04-01 15:24:12,605 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3338487].
2025-04-01 15:24:12,607 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.83585].
2025-04-01 15:24:12,611 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.35129].
2025-04-01 15:24:12,763 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.2991176].
2025-04-01 15:24:12,765 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.84265].
2025-04-01 15:24:12,768 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.36487].
2025-04-01 15:24:17,934 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:17,935 - utils.training - INFO - Global: J&F: 0.1172, J_mean: 0.2063, T_mean: 0.9911
2025-04-01 15:24:23,357 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:23,358 - utils.training - INFO - Global: J&F: 0.0951, J_mean: 0.1509, T_mean: 0.9920
2025-04-01 15:24:26,258 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0678318..2.1093895].
2025-04-01 15:24:26,261 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0678318..108.82069].
2025-04-01 15:24:26,264 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0678318..139.98984].
2025-04-01 15:24:26,397 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.067947..2.1309612].
2025-04-01 15:24:26,399 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.067947..108.83983].
2025-04-01 15:24:26,403 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.067947..139.94524].
2025-04-01 15:24:29,060 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:29,061 - utils.training - INFO - Global: J&F: 0.1082, J_mean: 0.1682, T_mean: 0.9849
2025-04-01 15:24:34,595 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:34,596 - utils.training - INFO - Global: J&F: 0.1164, J_mean: 0.1974, T_mean: 0.9814
2025-04-01 15:24:40,139 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:40,139 - utils.training - INFO - Global: J&F: 0.1313, J_mean: 0.2113, T_mean: 0.9853
2025-04-01 15:24:40,461 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7737776..2.455394].
2025-04-01 15:24:40,463 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7737776..108.96236].
2025-04-01 15:24:40,466 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7737776..153.75174].
2025-04-01 15:24:40,640 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7738038..2.4478843].
2025-04-01 15:24:40,642 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7738038..108.969635].
2025-04-01 15:24:40,645 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7738038..153.80492].
2025-04-01 15:24:45,957 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:45,958 - utils.training - INFO - Global: J&F: 0.1338, J_mean: 0.2198, T_mean: 0.9882
2025-04-01 15:24:51,411 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:51,411 - utils.training - INFO - Global: J&F: 0.1273, J_mean: 0.2154, T_mean: 0.9791
2025-04-01 15:24:53,884 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:24:53,884 - utils.training - INFO - Global: J&F: 0.1255, J_mean: 0.2004, T_mean: 0.9812
2025-04-01 15:24:53,981 - utils.training - INFO - Epoch 29 validation: val_loss: 1.2169 J&F: 0.1255 J_mean: 0.2004 F_mean: 0.0506 T_mean: 0.9812 instance_stability: 1.0000
2025-04-01 15:24:54,129 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 15:27:55,812 - utils.training - INFO - Epoch 30 completed: Loss: 1.4623, CE: 0.6871, Dice: 0.7752, LR: 0.000007
2025-04-01 15:30:57,438 - utils.training - INFO - Epoch 31 completed: Loss: 1.4620, CE: 0.6866, Dice: 0.7754, LR: 0.000009
2025-04-01 15:30:58,052 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.639966].
2025-04-01 15:30:58,054 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 15:30:58,059 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25598].
2025-04-01 15:30:58,291 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.639966].
2025-04-01 15:30:58,292 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055984].
2025-04-01 15:30:58,296 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25598].
2025-04-01 15:31:03,566 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:03,566 - utils.training - INFO - Global: J&F: 0.1832, J_mean: 0.3032, T_mean: 0.9873
2025-04-01 15:31:09,026 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:09,027 - utils.training - INFO - Global: J&F: 0.1401, J_mean: 0.2417, T_mean: 0.9919
2025-04-01 15:31:11,944 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.1798692].
2025-04-01 15:31:11,946 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.87194].
2025-04-01 15:31:11,949 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.87194].
2025-04-01 15:31:12,094 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.1802623].
2025-04-01 15:31:12,095 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.86958].
2025-04-01 15:31:12,099 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.87211].
2025-04-01 15:31:14,793 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:14,793 - utils.training - INFO - Global: J&F: 0.1359, J_mean: 0.2316, T_mean: 0.9944
2025-04-01 15:31:20,238 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:20,238 - utils.training - INFO - Global: J&F: 0.1658, J_mean: 0.2921, T_mean: 0.9917
2025-04-01 15:31:25,649 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:25,649 - utils.training - INFO - Global: J&F: 0.1151, J_mean: 0.1928, T_mean: 0.9804
2025-04-01 15:31:25,970 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7426554..2.2228389].
2025-04-01 15:31:25,973 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.741787..108.62624].
2025-04-01 15:31:25,976 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.741787..153.4088].
2025-04-01 15:31:26,123 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7427263..2.2238224].
2025-04-01 15:31:26,125 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7408208..108.676285].
2025-04-01 15:31:26,128 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7408208..153.42104].
2025-04-01 15:31:31,307 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:31,308 - utils.training - INFO - Global: J&F: 0.1098, J_mean: 0.1897, T_mean: 0.9916
2025-04-01 15:31:36,799 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:36,799 - utils.training - INFO - Global: J&F: 0.0938, J_mean: 0.1460, T_mean: 0.9924
2025-04-01 15:31:39,708 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3743403].
2025-04-01 15:31:39,710 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.943].
2025-04-01 15:31:39,714 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.1176].
2025-04-01 15:31:39,849 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.3841493].
2025-04-01 15:31:39,851 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.940575].
2025-04-01 15:31:39,854 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.04651].
2025-04-01 15:31:42,531 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:42,532 - utils.training - INFO - Global: J&F: 0.0922, J_mean: 0.1464, T_mean: 0.9892
2025-04-01 15:31:48,116 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:48,116 - utils.training - INFO - Global: J&F: 0.1152, J_mean: 0.1898, T_mean: 0.9846
2025-04-01 15:31:53,647 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:53,647 - utils.training - INFO - Global: J&F: 0.1537, J_mean: 0.2512, T_mean: 0.9863
2025-04-01 15:31:53,963 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9508877..2.5423062].
2025-04-01 15:31:53,965 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9508877..109.01692].
2025-04-01 15:31:53,969 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9508877..153.83813].
2025-04-01 15:31:54,132 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.950946..2.5422468].
2025-04-01 15:31:54,134 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.950946..109.0169].
2025-04-01 15:31:54,137 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.950946..153.86115].
2025-04-01 15:31:59,481 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:31:59,482 - utils.training - INFO - Global: J&F: 0.1487, J_mean: 0.2495, T_mean: 0.9913
2025-04-01 15:32:05,018 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:32:05,018 - utils.training - INFO - Global: J&F: 0.1288, J_mean: 0.2143, T_mean: 0.9808
2025-04-01 15:32:07,478 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:32:07,479 - utils.training - INFO - Global: J&F: 0.1278, J_mean: 0.2002, T_mean: 0.9822
2025-04-01 15:32:07,559 - utils.training - INFO - Epoch 31 validation: val_loss: 1.2673 J&F: 0.1278 J_mean: 0.2002 F_mean: 0.0555 T_mean: 0.9822 instance_stability: 1.0000
2025-04-01 15:35:09,310 - utils.training - INFO - Epoch 32 completed: Loss: 1.4620, CE: 0.6866, Dice: 0.7754, LR: 0.000011
2025-04-01 15:38:10,660 - utils.training - INFO - Epoch 33 completed: Loss: 1.4623, CE: 0.6871, Dice: 0.7753, LR: 0.000014
2025-04-01 15:38:11,244 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8497546..1.8819898].
2025-04-01 15:38:11,247 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8497546..108.75244].
2025-04-01 15:38:11,252 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8497546..139.89735].
2025-04-01 15:38:11,470 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8493105..1.8824418].
2025-04-01 15:38:11,472 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8493105..108.752975].
2025-04-01 15:38:11,475 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8493105..139.90312].
2025-04-01 15:38:16,703 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:16,704 - utils.training - INFO - Global: J&F: 0.1727, J_mean: 0.2898, T_mean: 0.9869
2025-04-01 15:38:22,195 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:22,196 - utils.training - INFO - Global: J&F: 0.1471, J_mean: 0.2575, T_mean: 0.9908
2025-04-01 15:38:25,101 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9060925..1.8617431].
2025-04-01 15:38:25,104 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9060925..108.7447].
2025-04-01 15:38:25,108 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8844597..108.73993].
2025-04-01 15:38:25,255 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8821193..1.8560528].
2025-04-01 15:38:25,256 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8747482..108.742424].
2025-04-01 15:38:25,260 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8821193..108.739136].
2025-04-01 15:38:27,917 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:27,918 - utils.training - INFO - Global: J&F: 0.1674, J_mean: 0.2950, T_mean: 0.9953
2025-04-01 15:38:33,365 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:33,366 - utils.training - INFO - Global: J&F: 0.1632, J_mean: 0.2854, T_mean: 0.9907
2025-04-01 15:38:38,814 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:38,814 - utils.training - INFO - Global: J&F: 0.1349, J_mean: 0.2346, T_mean: 0.9803
2025-04-01 15:38:39,137 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:38:39,138 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92417].
2025-04-01 15:38:39,142 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.71309].
2025-04-01 15:38:39,284 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:38:39,285 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.92319].
2025-04-01 15:38:39,288 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.73088].
2025-04-01 15:38:44,456 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:44,457 - utils.training - INFO - Global: J&F: 0.1204, J_mean: 0.2121, T_mean: 0.9932
2025-04-01 15:38:49,849 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:49,849 - utils.training - INFO - Global: J&F: 0.1397, J_mean: 0.2159, T_mean: 0.9918
2025-04-01 15:38:52,786 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.450633].
2025-04-01 15:38:52,788 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.97366].
2025-04-01 15:38:52,792 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.14427].
2025-04-01 15:38:52,924 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.483188].
2025-04-01 15:38:52,926 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.97965].
2025-04-01 15:38:52,929 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.08548].
2025-04-01 15:38:55,562 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:38:55,563 - utils.training - INFO - Global: J&F: 0.1172, J_mean: 0.1867, T_mean: 0.9870
2025-04-01 15:39:01,182 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:39:01,182 - utils.training - INFO - Global: J&F: 0.1201, J_mean: 0.2059, T_mean: 0.9845
2025-04-01 15:39:06,717 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:39:06,718 - utils.training - INFO - Global: J&F: 0.1715, J_mean: 0.2724, T_mean: 0.9838
2025-04-01 15:39:07,076 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:39:07,078 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:39:07,081 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89957].
2025-04-01 15:39:07,251 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:39:07,253 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:39:07,256 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.89957].
2025-04-01 15:39:12,562 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:39:12,562 - utils.training - INFO - Global: J&F: 0.1378, J_mean: 0.2264, T_mean: 0.9888
2025-04-01 15:39:18,047 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:39:18,048 - utils.training - INFO - Global: J&F: 0.1339, J_mean: 0.2248, T_mean: 0.9808
2025-04-01 15:39:20,488 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:39:20,489 - utils.training - INFO - Global: J&F: 0.1303, J_mean: 0.2084, T_mean: 0.9836
2025-04-01 15:39:20,601 - utils.training - INFO - Epoch 33 validation: val_loss: 1.2427 J&F: 0.1303 J_mean: 0.2084 F_mean: 0.0522 T_mean: 0.9836 instance_stability: 1.0000
2025-04-01 15:42:22,372 - utils.training - INFO - Epoch 34 completed: Loss: 1.4629, CE: 0.6871, Dice: 0.7759, LR: 0.000016
2025-04-01 15:45:23,997 - utils.training - INFO - Epoch 35 completed: Loss: 1.4608, CE: 0.6863, Dice: 0.7745, LR: 0.000019
2025-04-01 15:45:24,590 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8675444..2.1809778].
2025-04-01 15:45:24,593 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8675444..108.87239].
2025-04-01 15:45:24,599 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8675444..140.07239].
2025-04-01 15:45:24,813 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8671545..2.1813748].
2025-04-01 15:45:24,814 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8671545..108.87255].
2025-04-01 15:45:24,817 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8671545..140.07256].
2025-04-01 15:45:30,039 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:45:30,039 - utils.training - INFO - Global: J&F: 0.1624, J_mean: 0.2603, T_mean: 0.9869
2025-04-01 15:45:35,539 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:45:35,539 - utils.training - INFO - Global: J&F: 0.1680, J_mean: 0.2933, T_mean: 0.9897
2025-04-01 15:45:38,478 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.639998].
2025-04-01 15:45:38,480 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:45:38,483 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:45:38,649 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.639998].
2025-04-01 15:45:38,651 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:45:38,654 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:45:41,288 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:45:41,289 - utils.training - INFO - Global: J&F: 0.1846, J_mean: 0.3271, T_mean: 0.9939
2025-04-01 15:45:46,714 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:45:46,714 - utils.training - INFO - Global: J&F: 0.1684, J_mean: 0.2922, T_mean: 0.9891
2025-04-01 15:45:52,156 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:45:52,157 - utils.training - INFO - Global: J&F: 0.1604, J_mean: 0.2667, T_mean: 0.9819
2025-04-01 15:45:52,476 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.9076914].
2025-04-01 15:45:52,478 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.72494].
2025-04-01 15:45:52,481 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.2751].
2025-04-01 15:45:52,649 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.9078262].
2025-04-01 15:45:52,651 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.744255].
2025-04-01 15:45:52,654 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.28714].
2025-04-01 15:45:57,791 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:45:57,792 - utils.training - INFO - Global: J&F: 0.1466, J_mean: 0.2650, T_mean: 0.9896
2025-04-01 15:46:03,189 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:03,189 - utils.training - INFO - Global: J&F: 0.1063, J_mean: 0.1686, T_mean: 0.9918
2025-04-01 15:46:06,103 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:46:06,106 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:46:06,109 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 15:46:06,248 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 15:46:06,250 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 15:46:06,253 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 15:46:08,913 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:08,913 - utils.training - INFO - Global: J&F: 0.0881, J_mean: 0.1475, T_mean: 0.9847
2025-04-01 15:46:14,459 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:14,459 - utils.training - INFO - Global: J&F: 0.1304, J_mean: 0.2138, T_mean: 0.9817
2025-04-01 15:46:20,034 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:20,034 - utils.training - INFO - Global: J&F: 0.1712, J_mean: 0.2692, T_mean: 0.9855
2025-04-01 15:46:20,364 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6378834].
2025-04-01 15:46:20,366 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05333].
2025-04-01 15:46:20,370 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.76071].
2025-04-01 15:46:20,589 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.636746].
2025-04-01 15:46:20,591 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.033875].
2025-04-01 15:46:20,594 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.85866].
2025-04-01 15:46:25,922 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:25,923 - utils.training - INFO - Global: J&F: 0.1655, J_mean: 0.2711, T_mean: 0.9887
2025-04-01 15:46:31,439 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:31,439 - utils.training - INFO - Global: J&F: 0.1362, J_mean: 0.2303, T_mean: 0.9804
2025-04-01 15:46:33,893 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:46:33,894 - utils.training - INFO - Global: J&F: 0.1213, J_mean: 0.1909, T_mean: 0.9785
2025-04-01 15:46:33,972 - utils.training - INFO - Epoch 35 validation: val_loss: 1.1982 J&F: 0.1213 J_mean: 0.1909 F_mean: 0.0517 T_mean: 0.9785 instance_stability: 1.0000
2025-04-01 15:46:34,145 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 15:49:35,570 - utils.training - INFO - Epoch 36 completed: Loss: 1.4622, CE: 0.6870, Dice: 0.7751, LR: 0.000022
2025-04-01 15:52:37,378 - utils.training - INFO - Epoch 37 completed: Loss: 1.4623, CE: 0.6872, Dice: 0.7751, LR: 0.000025
2025-04-01 15:52:37,956 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.46503].
2025-04-01 15:52:37,960 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.9859].
2025-04-01 15:52:37,970 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.1158].
2025-04-01 15:52:38,181 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4646888].
2025-04-01 15:52:38,183 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.98588].
2025-04-01 15:52:38,186 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.12292].
2025-04-01 15:52:43,450 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:52:43,450 - utils.training - INFO - Global: J&F: 0.2093, J_mean: 0.3442, T_mean: 0.9876
2025-04-01 15:52:48,930 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:52:48,930 - utils.training - INFO - Global: J&F: 0.1761, J_mean: 0.3089, T_mean: 0.9898
2025-04-01 15:52:51,848 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8718208..1.9155242].
2025-04-01 15:52:51,850 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8718208..108.76621].
2025-04-01 15:52:51,853 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8718208..108.76621].
2025-04-01 15:52:52,015 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8710254..1.916334].
2025-04-01 15:52:52,017 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8710254..108.76653].
2025-04-01 15:52:52,020 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8710254..108.76653].
2025-04-01 15:52:54,686 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:52:54,686 - utils.training - INFO - Global: J&F: 0.1545, J_mean: 0.2735, T_mean: 0.9939
2025-04-01 15:53:00,101 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:00,101 - utils.training - INFO - Global: J&F: 0.1686, J_mean: 0.2895, T_mean: 0.9895
2025-04-01 15:53:05,548 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:05,548 - utils.training - INFO - Global: J&F: 0.1389, J_mean: 0.2417, T_mean: 0.9815
2025-04-01 15:53:05,869 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9170909..1.9143723].
2025-04-01 15:53:05,872 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9162996..108.64013].
2025-04-01 15:53:05,875 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.915637..153.31271].
2025-04-01 15:53:06,045 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9165092..1.9142905].
2025-04-01 15:53:06,046 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9165092..108.646736].
2025-04-01 15:53:06,050 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9150553..153.32451].
2025-04-01 15:53:11,208 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:11,208 - utils.training - INFO - Global: J&F: 0.1414, J_mean: 0.2516, T_mean: 0.9924
2025-04-01 15:53:16,608 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:16,608 - utils.training - INFO - Global: J&F: 0.0874, J_mean: 0.1431, T_mean: 0.9921
2025-04-01 15:53:19,514 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399863].
2025-04-01 15:53:19,516 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.055504].
2025-04-01 15:53:19,518 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25136].
2025-04-01 15:53:19,659 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399863].
2025-04-01 15:53:19,660 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.0549].
2025-04-01 15:53:19,663 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.24886].
2025-04-01 15:53:22,333 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:22,333 - utils.training - INFO - Global: J&F: 0.1005, J_mean: 0.1635, T_mean: 0.9875
2025-04-01 15:53:27,854 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:27,855 - utils.training - INFO - Global: J&F: 0.1260, J_mean: 0.2119, T_mean: 0.9810
2025-04-01 15:53:33,384 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:33,385 - utils.training - INFO - Global: J&F: 0.1553, J_mean: 0.2570, T_mean: 0.9872
2025-04-01 15:53:33,717 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7353653..2.0731452].
2025-04-01 15:53:33,719 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7353653..108.82926].
2025-04-01 15:53:33,722 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7353653..153.5566].
2025-04-01 15:53:33,903 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7354397..2.071254].
2025-04-01 15:53:33,905 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7354397..108.8285].
2025-04-01 15:53:33,909 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.7354397..153.62514].
2025-04-01 15:53:39,227 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:39,228 - utils.training - INFO - Global: J&F: 0.1349, J_mean: 0.2224, T_mean: 0.9881
2025-04-01 15:53:44,655 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:44,655 - utils.training - INFO - Global: J&F: 0.1382, J_mean: 0.2291, T_mean: 0.9841
2025-04-01 15:53:47,048 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:53:47,048 - utils.training - INFO - Global: J&F: 0.1185, J_mean: 0.1856, T_mean: 0.9800
2025-04-01 15:53:47,128 - utils.training - INFO - Epoch 37 validation: val_loss: 1.1934 J&F: 0.1185 J_mean: 0.1856 F_mean: 0.0514 T_mean: 0.9800 instance_stability: 1.0000
2025-04-01 15:53:47,298 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 15:56:49,239 - utils.training - INFO - Epoch 38 completed: Loss: 1.4615, CE: 0.6870, Dice: 0.7745, LR: 0.000029
2025-04-01 15:59:49,828 - utils.training - INFO - Epoch 39 completed: Loss: 1.4602, CE: 0.6866, Dice: 0.7736, LR: 0.000032
2025-04-01 15:59:50,400 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0384996..1.9122946].
2025-04-01 15:59:50,404 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0384996..108.76469].
2025-04-01 15:59:50,409 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0384996..139.89954].
2025-04-01 15:59:50,623 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0383694..1.9124268].
2025-04-01 15:59:50,625 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0383694..108.76497].
2025-04-01 15:59:50,628 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0383694..139.90575].
2025-04-01 15:59:55,833 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 15:59:55,833 - utils.training - INFO - Global: J&F: 0.1884, J_mean: 0.3136, T_mean: 0.9864
2025-04-01 16:00:01,292 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:01,292 - utils.training - INFO - Global: J&F: 0.1555, J_mean: 0.2713, T_mean: 0.9905
2025-04-01 16:00:04,210 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0345805..2.5348701].
2025-04-01 16:00:04,212 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0345805..109.01297].
2025-04-01 16:00:04,215 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0145206..109.01395].
2025-04-01 16:00:04,380 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.012239..2.5337808].
2025-04-01 16:00:04,381 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0076594..109.01065].
2025-04-01 16:00:04,385 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.012239..109.01172].
2025-04-01 16:00:07,033 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:07,033 - utils.training - INFO - Global: J&F: 0.1866, J_mean: 0.3319, T_mean: 0.9947
2025-04-01 16:00:12,466 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:12,466 - utils.training - INFO - Global: J&F: 0.1815, J_mean: 0.3101, T_mean: 0.9899
2025-04-01 16:00:17,942 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:17,942 - utils.training - INFO - Global: J&F: 0.1505, J_mean: 0.2442, T_mean: 0.9820
2025-04-01 16:00:18,260 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4573257].
2025-04-01 16:00:18,263 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.89003].
2025-04-01 16:00:18,267 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.38486].
2025-04-01 16:00:18,415 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.4388006].
2025-04-01 16:00:18,417 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.91169].
2025-04-01 16:00:18,421 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.39877].
2025-04-01 16:00:23,572 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:23,573 - utils.training - INFO - Global: J&F: 0.1417, J_mean: 0.2527, T_mean: 0.9929
2025-04-01 16:00:28,985 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:28,985 - utils.training - INFO - Global: J&F: 0.0957, J_mean: 0.1504, T_mean: 0.9914
2025-04-01 16:00:31,892 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399448].
2025-04-01 16:00:31,894 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05598].
2025-04-01 16:00:31,897 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.24744].
2025-04-01 16:00:32,036 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6399448].
2025-04-01 16:00:32,037 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05598].
2025-04-01 16:00:32,040 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.25598].
2025-04-01 16:00:34,671 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:34,671 - utils.training - INFO - Global: J&F: 0.1027, J_mean: 0.1643, T_mean: 0.9870
2025-04-01 16:00:40,188 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:40,189 - utils.training - INFO - Global: J&F: 0.1345, J_mean: 0.2255, T_mean: 0.9833
2025-04-01 16:00:45,720 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:45,720 - utils.training - INFO - Global: J&F: 0.1491, J_mean: 0.2396, T_mean: 0.9855
2025-04-01 16:00:46,041 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.956552].
2025-04-01 16:00:46,044 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.78262].
2025-04-01 16:00:46,048 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.49449].
2025-04-01 16:00:46,208 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..1.9544058].
2025-04-01 16:00:46,210 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.78176].
2025-04-01 16:00:46,213 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.57446].
2025-04-01 16:00:51,521 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:51,522 - utils.training - INFO - Global: J&F: 0.1789, J_mean: 0.2948, T_mean: 0.9904
2025-04-01 16:00:57,024 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:57,025 - utils.training - INFO - Global: J&F: 0.1411, J_mean: 0.2328, T_mean: 0.9772
2025-04-01 16:00:59,474 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:00:59,474 - utils.training - INFO - Global: J&F: 0.1248, J_mean: 0.1972, T_mean: 0.9818
2025-04-01 16:00:59,559 - utils.training - INFO - Epoch 39 validation: val_loss: 1.1916 J&F: 0.1248 J_mean: 0.1972 F_mean: 0.0525 T_mean: 0.9818 instance_stability: 1.0000
2025-04-01 16:00:59,732 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 16:00:59,905 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 16:03:59,158 - utils.training - INFO - Epoch 40 completed: Loss: 1.4604, CE: 0.6865, Dice: 0.7739, LR: 0.000035
2025-04-01 16:06:56,919 - utils.training - INFO - Epoch 41 completed: Loss: 1.4585, CE: 0.6858, Dice: 0.7728, LR: 0.000037
2025-04-01 16:06:57,508 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:06:57,514 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 16:06:57,522 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 16:06:57,741 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:06:57,743 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 16:06:57,747 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.256].
2025-04-01 16:07:02,897 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:02,897 - utils.training - INFO - Global: J&F: 0.1709, J_mean: 0.2830, T_mean: 0.9880
2025-04-01 16:07:08,348 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:08,349 - utils.training - INFO - Global: J&F: 0.1704, J_mean: 0.2917, T_mean: 0.9890
2025-04-01 16:07:11,240 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0897927..1.9898318].
2025-04-01 16:07:11,241 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0897927..108.79593].
2025-04-01 16:07:11,244 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0897927..108.79592].
2025-04-01 16:07:11,396 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.089717..1.9899089].
2025-04-01 16:07:11,399 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.089717..108.794495].
2025-04-01 16:07:11,402 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.089717..108.79597].
2025-04-01 16:07:14,065 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:14,065 - utils.training - INFO - Global: J&F: 0.1949, J_mean: 0.3411, T_mean: 0.9945
2025-04-01 16:07:19,455 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:19,455 - utils.training - INFO - Global: J&F: 0.1679, J_mean: 0.2814, T_mean: 0.9896
2025-04-01 16:07:24,905 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:24,906 - utils.training - INFO - Global: J&F: 0.1553, J_mean: 0.2505, T_mean: 0.9857
2025-04-01 16:07:25,221 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8507901..1.9717258].
2025-04-01 16:07:25,224 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.850461..108.74821].
2025-04-01 16:07:25,227 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8501854..153.31886].
2025-04-01 16:07:25,377 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.850048..1.9580215].
2025-04-01 16:07:25,378 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8497188..108.73694].
2025-04-01 16:07:25,382 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8494432..153.33046].
2025-04-01 16:07:30,466 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:30,467 - utils.training - INFO - Global: J&F: 0.1599, J_mean: 0.2840, T_mean: 0.9930
2025-04-01 16:07:35,850 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:35,850 - utils.training - INFO - Global: J&F: 0.1197, J_mean: 0.1884, T_mean: 0.9948
2025-04-01 16:07:38,742 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9821106..2.4751267].
2025-04-01 16:07:38,744 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9821106..108.98468].
2025-04-01 16:07:38,747 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9821106..140.18048].
2025-04-01 16:07:38,891 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9824327..2.477267].
2025-04-01 16:07:38,893 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9824327..108.980286].
2025-04-01 16:07:38,896 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9824327..140.18166].
2025-04-01 16:07:41,558 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:41,558 - utils.training - INFO - Global: J&F: 0.1055, J_mean: 0.1685, T_mean: 0.9894
2025-04-01 16:07:47,070 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:47,071 - utils.training - INFO - Global: J&F: 0.1161, J_mean: 0.1921, T_mean: 0.9827
2025-04-01 16:07:52,556 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:52,556 - utils.training - INFO - Global: J&F: 0.1675, J_mean: 0.2686, T_mean: 0.9871
2025-04-01 16:07:52,875 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6366684].
2025-04-01 16:07:52,877 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.054665].
2025-04-01 16:07:52,880 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.75737].
2025-04-01 16:07:53,052 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.6385663].
2025-04-01 16:07:53,053 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.05543].
2025-04-01 16:07:53,057 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.85628].
2025-04-01 16:07:58,336 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:07:58,337 - utils.training - INFO - Global: J&F: 0.1775, J_mean: 0.2915, T_mean: 0.9916
2025-04-01 16:08:03,794 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:08:03,794 - utils.training - INFO - Global: J&F: 0.1501, J_mean: 0.2483, T_mean: 0.9840
2025-04-01 16:08:06,263 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:08:06,264 - utils.training - INFO - Global: J&F: 0.1302, J_mean: 0.2021, T_mean: 0.9819
2025-04-01 16:08:06,343 - utils.training - INFO - Epoch 41 validation: val_loss: 1.2031 J&F: 0.1302 J_mean: 0.2021 F_mean: 0.0583 T_mean: 0.9819 instance_stability: 1.0000
2025-04-01 16:11:04,321 - utils.training - INFO - Epoch 42 completed: Loss: 1.4581, CE: 0.6856, Dice: 0.7725, LR: 0.000040
2025-04-01 16:13:59,316 - utils.training - INFO - Epoch 43 completed: Loss: 1.4579, CE: 0.6860, Dice: 0.7720, LR: 0.000042
2025-04-01 16:13:59,956 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8903694..1.3901304].
2025-04-01 16:13:59,963 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8903694..108.55321].
2025-04-01 16:13:59,970 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8903694..139.71828].
2025-04-01 16:14:00,178 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8900058..1.3911014].
2025-04-01 16:14:00,180 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8900058..108.55384].
2025-04-01 16:14:00,184 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8900058..139.72282].
2025-04-01 16:14:05,410 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:05,411 - utils.training - INFO - Global: J&F: 0.1544, J_mean: 0.2461, T_mean: 0.9826
2025-04-01 16:14:10,840 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:10,840 - utils.training - INFO - Global: J&F: 0.1835, J_mean: 0.3172, T_mean: 0.9863
2025-04-01 16:14:13,710 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.0419383].
2025-04-01 16:14:13,712 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.81677].
2025-04-01 16:14:13,715 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.79222].
2025-04-01 16:14:13,876 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.0366824].
2025-04-01 16:14:13,878 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0681813..108.814674].
2025-04-01 16:14:13,881 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.789116].
2025-04-01 16:14:16,508 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:16,508 - utils.training - INFO - Global: J&F: 0.2260, J_mean: 0.3978, T_mean: 0.9933
2025-04-01 16:14:21,852 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:21,852 - utils.training - INFO - Global: J&F: 0.2002, J_mean: 0.3433, T_mean: 0.9878
2025-04-01 16:14:27,295 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:27,296 - utils.training - INFO - Global: J&F: 0.1317, J_mean: 0.2231, T_mean: 0.9796
2025-04-01 16:14:27,608 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:14:27,611 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.8427].
2025-04-01 16:14:27,615 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.54211].
2025-04-01 16:14:27,754 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:14:27,755 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..108.84815].
2025-04-01 16:14:27,758 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.55775].
2025-04-01 16:14:32,871 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:32,871 - utils.training - INFO - Global: J&F: 0.1221, J_mean: 0.2152, T_mean: 0.9896
2025-04-01 16:14:38,237 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:38,238 - utils.training - INFO - Global: J&F: 0.1043, J_mean: 0.1649, T_mean: 0.9907
2025-04-01 16:14:41,112 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0239031..1.9806511].
2025-04-01 16:14:41,114 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0239031..108.79226].
2025-04-01 16:14:41,117 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0239031..139.9193].
2025-04-01 16:14:41,250 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0241728..1.9979417].
2025-04-01 16:14:41,252 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0241728..108.78876].
2025-04-01 16:14:41,255 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0241728..139.88698].
2025-04-01 16:14:43,913 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:43,913 - utils.training - INFO - Global: J&F: 0.1109, J_mean: 0.1798, T_mean: 0.9844
2025-04-01 16:14:49,400 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:49,401 - utils.training - INFO - Global: J&F: 0.1223, J_mean: 0.2016, T_mean: 0.9782
2025-04-01 16:14:54,891 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:14:54,891 - utils.training - INFO - Global: J&F: 0.1421, J_mean: 0.2268, T_mean: 0.9837
2025-04-01 16:14:55,209 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9267478..2.5367684].
2025-04-01 16:14:55,211 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9267478..109.01471].
2025-04-01 16:14:55,215 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9267478..153.859].
2025-04-01 16:14:55,370 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9268149..2.5366995].
2025-04-01 16:14:55,372 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9268149..109.01468].
2025-04-01 16:14:55,375 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9268149..153.85896].
2025-04-01 16:15:00,646 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:15:00,647 - utils.training - INFO - Global: J&F: 0.1481, J_mean: 0.2435, T_mean: 0.9867
2025-04-01 16:15:06,093 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:15:06,093 - utils.training - INFO - Global: J&F: 0.1465, J_mean: 0.2443, T_mean: 0.9803
2025-04-01 16:15:08,535 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:15:08,535 - utils.training - INFO - Global: J&F: 0.1253, J_mean: 0.2010, T_mean: 0.9785
2025-04-01 16:15:08,611 - utils.training - INFO - Epoch 43 validation: val_loss: 1.1419 J&F: 0.1253 J_mean: 0.2010 F_mean: 0.0496 T_mean: 0.9785 instance_stability: 1.0000
2025-04-01 16:15:08,784 - utils.training - INFO - Saved checkpoint and metrics to checkpoints/rtx4070ti_super_fast
2025-04-01 16:18:03,707 - utils.training - INFO - Epoch 44 completed: Loss: 1.4559, CE: 0.6848, Dice: 0.7711, LR: 0.000044
2025-04-01 16:20:58,707 - utils.training - INFO - Epoch 45 completed: Loss: 1.4543, CE: 0.6845, Dice: 0.7698, LR: 0.000046
2025-04-01 16:20:59,308 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.00888..2.4463522].
2025-04-01 16:20:59,312 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.00888..108.97841].
2025-04-01 16:20:59,319 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.00888..140.17848].
2025-04-01 16:20:59,520 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0087252..2.4464772].
2025-04-01 16:20:59,521 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0087252..108.978485].
2025-04-01 16:20:59,525 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0087252..140.17859].
2025-04-01 16:21:04,708 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:04,708 - utils.training - INFO - Global: J&F: 0.1517, J_mean: 0.2435, T_mean: 0.9894
2025-04-01 16:21:10,146 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:10,147 - utils.training - INFO - Global: J&F: 0.1780, J_mean: 0.3069, T_mean: 0.9898
2025-04-01 16:21:13,022 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.046863..1.8516262].
2025-04-01 16:21:13,023 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0102935..108.740654].
2025-04-01 16:21:13,026 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0023258..108.72307].
2025-04-01 16:21:13,185 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9994178..1.8455253].
2025-04-01 16:21:13,187 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9677235..108.73821].
2025-04-01 16:21:13,190 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9994178..108.7211].
2025-04-01 16:21:15,812 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:15,813 - utils.training - INFO - Global: J&F: 0.1914, J_mean: 0.3313, T_mean: 0.9946
2025-04-01 16:21:21,165 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:21,166 - utils.training - INFO - Global: J&F: 0.2066, J_mean: 0.3524, T_mean: 0.9921
2025-04-01 16:21:26,554 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:26,554 - utils.training - INFO - Global: J&F: 0.1572, J_mean: 0.2589, T_mean: 0.9814
2025-04-01 16:21:26,865 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:21:26,868 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 16:21:26,871 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.63567].
2025-04-01 16:21:27,006 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:21:27,008 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 16:21:27,011 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..153.65242].
2025-04-01 16:21:32,101 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:32,101 - utils.training - INFO - Global: J&F: 0.1496, J_mean: 0.2643, T_mean: 0.9925
2025-04-01 16:21:37,392 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:37,393 - utils.training - INFO - Global: J&F: 0.0949, J_mean: 0.1464, T_mean: 0.9905
2025-04-01 16:21:40,265 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:21:40,268 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 16:21:40,271 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.22905].
2025-04-01 16:21:40,411 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..2.64].
2025-04-01 16:21:40,413 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..109.056].
2025-04-01 16:21:40,417 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.117904..140.22997].
2025-04-01 16:21:43,071 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:43,072 - utils.training - INFO - Global: J&F: 0.0874, J_mean: 0.1455, T_mean: 0.9885
2025-04-01 16:21:48,509 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:48,509 - utils.training - INFO - Global: J&F: 0.1265, J_mean: 0.2115, T_mean: 0.9836
2025-04-01 16:21:53,953 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:53,954 - utils.training - INFO - Global: J&F: 0.1414, J_mean: 0.2334, T_mean: 0.9876
2025-04-01 16:21:54,266 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.069431..2.6096404].
2025-04-01 16:21:54,269 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.069431..109.04385].
2025-04-01 16:21:54,272 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.069431..153.87152].
2025-04-01 16:21:54,435 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0694416..2.6096296].
2025-04-01 16:21:54,436 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0694416..109.04385].
2025-04-01 16:21:54,440 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0694416..153.84938].
2025-04-01 16:21:59,693 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:21:59,694 - utils.training - INFO - Global: J&F: 0.1270, J_mean: 0.2128, T_mean: 0.9888
2025-04-01 16:22:05,103 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:22:05,103 - utils.training - INFO - Global: J&F: 0.1330, J_mean: 0.2262, T_mean: 0.9833
2025-04-01 16:22:07,524 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:22:07,524 - utils.training - INFO - Global: J&F: 0.1461, J_mean: 0.2289, T_mean: 0.9826
2025-04-01 16:22:07,601 - utils.training - INFO - Epoch 45 validation: val_loss: 1.1835 J&F: 0.1461 J_mean: 0.2289 F_mean: 0.0632 T_mean: 0.9826 instance_stability: 1.0000
2025-04-01 16:25:02,516 - utils.training - INFO - Epoch 46 completed: Loss: 1.4546, CE: 0.6844, Dice: 0.7702, LR: 0.000048
2025-04-01 16:27:57,478 - utils.training - INFO - Epoch 47 completed: Loss: 1.4537, CE: 0.6841, Dice: 0.7696, LR: 0.000049
2025-04-01 16:27:58,105 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9480139..1.8201933].
2025-04-01 16:27:58,108 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9480139..108.72805].
2025-04-01 16:27:58,116 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9480139..139.8788].
2025-04-01 16:27:58,340 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.94775..1.8204622].
2025-04-01 16:27:58,342 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.94775..108.728165].
2025-04-01 16:27:58,345 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.94775..139.88434].
2025-04-01 16:28:03,507 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:03,507 - utils.training - INFO - Global: J&F: 0.1219, J_mean: 0.1937, T_mean: 0.9876
2025-04-01 16:28:08,894 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:08,894 - utils.training - INFO - Global: J&F: 0.1845, J_mean: 0.3290, T_mean: 0.9921
2025-04-01 16:28:11,764 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9131399..1.8098025].
2025-04-01 16:28:11,766 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9131399..108.723854].
2025-04-01 16:28:11,769 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9131399..108.723915].
2025-04-01 16:28:11,926 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.91255..1.8104028].
2025-04-01 16:28:11,928 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.91255..108.7241].
2025-04-01 16:28:11,931 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.91255..108.72416].
2025-04-01 16:28:14,537 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:14,537 - utils.training - INFO - Global: J&F: 0.1390, J_mean: 0.2402, T_mean: 0.9950
2025-04-01 16:28:19,904 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:19,905 - utils.training - INFO - Global: J&F: 0.2021, J_mean: 0.3535, T_mean: 0.9925
2025-04-01 16:28:25,201 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:25,201 - utils.training - INFO - Global: J&F: 0.1617, J_mean: 0.2732, T_mean: 0.9806
2025-04-01 16:28:25,528 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8317348..1.5682787].
2025-04-01 16:28:25,531 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8317348..108.60285].
2025-04-01 16:28:25,534 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8317348..153.17761].
2025-04-01 16:28:25,679 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8309923..1.5567776].
2025-04-01 16:28:25,681 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8309923..108.59827].
2025-04-01 16:28:25,685 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8309923..153.18759].
2025-04-01 16:28:30,807 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:30,808 - utils.training - INFO - Global: J&F: 0.1195, J_mean: 0.2142, T_mean: 0.9943
2025-04-01 16:28:36,152 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:36,153 - utils.training - INFO - Global: J&F: 0.0665, J_mean: 0.1099, T_mean: 0.9881
2025-04-01 16:28:39,049 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.866957..2.0983386].
2025-04-01 16:28:39,051 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.866957..108.83377].
2025-04-01 16:28:39,054 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.866957..139.96036].
2025-04-01 16:28:39,192 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8675385..2.111427].
2025-04-01 16:28:39,194 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8675385..108.82605].
2025-04-01 16:28:39,196 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.8675385..139.94292].
2025-04-01 16:28:41,856 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:41,856 - utils.training - INFO - Global: J&F: 0.0941, J_mean: 0.1572, T_mean: 0.9882
2025-04-01 16:28:47,371 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:47,372 - utils.training - INFO - Global: J&F: 0.1233, J_mean: 0.2128, T_mean: 0.9866
2025-04-01 16:28:52,819 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:52,820 - utils.training - INFO - Global: J&F: 0.1448, J_mean: 0.2326, T_mean: 0.9888
2025-04-01 16:28:53,144 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.980262..2.058327].
2025-04-01 16:28:53,146 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.980262..108.82333].
2025-04-01 16:28:53,149 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.980262..153.54231].
2025-04-01 16:28:53,337 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9803144..2.05598].
2025-04-01 16:28:53,339 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9803144..108.822395].
2025-04-01 16:28:53,342 - matplotlib.image - WARNING - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.9803144..153.61732].
2025-04-01 16:28:58,633 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:28:58,634 - utils.training - INFO - Global: J&F: 0.1286, J_mean: 0.2134, T_mean: 0.9898
2025-04-01 16:29:04,069 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:29:04,069 - utils.training - INFO - Global: J&F: 0.1167, J_mean: 0.1985, T_mean: 0.9843
2025-04-01 16:29:06,560 - utils.training - INFO - 
Partial Evaluation Results:
2025-04-01 16:29:06,560 - utils.training - INFO - Global: J&F: 0.1077, J_mean: 0.1648, T_mean: 0.9876
2025-04-01 16:29:06,651 - utils.training - INFO - Epoch 47 validation: val_loss: 1.2791 J&F: 0.1077 J_mean: 0.1648 F_mean: 0.0507 T_mean: 0.9876 instance_stability: 1.0000

```

# logs\rtx4070ti_super\training_20250331_205611.log

```log
2025-03-31 20:56:11,290 - __main__ - INFO - Starting binary video segmentation training with config: configs/rtx_4070ti_super.yaml
2025-03-31 20:56:11,294 - __main__ - INFO - Set random seed to 42
2025-03-31 20:56:11,686 - __main__ - INFO - Using device: cuda
2025-03-31 20:56:11,687 - __main__ - INFO - Building binary segmentation model...
2025-03-31 20:56:12,183 - __main__ - INFO - Model created: VideoMambaSegmentation
2025-03-31 20:56:12,184 - __main__ - INFO - Created data augmentation with image size: [320, 384]
2025-03-31 20:56:12,184 - __main__ - INFO - Creating train data loader...
2025-03-31 20:56:12,617 - __main__ - INFO - Train loader created with 1028 batches
2025-03-31 20:56:12,617 - __main__ - INFO - Creating validation data loader...
2025-03-31 20:56:12,895 - __main__ - INFO - Validation loader created with 487 batches
2025-03-31 20:56:12,895 - __main__ - INFO - Creating optimizer: AdamW
2025-03-31 20:56:12,896 - __main__ - INFO - Creating scheduler: cosine
2025-03-31 20:56:12,896 - __main__ - INFO - Created BinarySegmentationLoss with weights: CE=1.0, Dice=1.0
2025-03-31 20:56:12,900 - __main__ - INFO - Trainer initialized
2025-03-31 20:56:12,900 - __main__ - INFO - Starting training process...
2025-03-31 20:56:12,901 - utils.training - INFO - Starting training from epoch 0

```

# losses\__init__.py

```py
# losses/__init__.py

from .temporal_consistency import TemporalConsistencyLoss
from .video_instance_loss import (
    VideoInstanceSegmentationLoss,
    DiceLoss,
    FocalLoss
)

__all__ = [
    'TemporalConsistencyLoss',
    'VideoInstanceSegmentationLoss',
    'DiceLoss',
    'FocalLoss'
]
```

# losses\combined.py

```py
import torch
import torch.nn as nn
from typing import Dict, Optional


from .segmentation import BinarySegmentationLoss
from .temporal_consistency import TemporalConsistencyLoss

class CombinedLoss(nn.Module):
    """
    Combines binary segmentation loss with temporal consistency loss.
    This is designed for binary video segmentation rather than instance segmentation.
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        temporal_weight: float = 1.0
    ):
        super().__init__()
        self.seg_loss = BinarySegmentationLoss(ce_weight, dice_weight)
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        flows: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with proper temporal handling.
        
        Args:
            outputs: Dictionary containing 'logits' or 'pred_masks'
            targets: Dictionary containing 'masks'
            flows: Optional optical flow between frames
            
        Returns:
            Dictionary containing all loss terms and total loss
        """
        # Compute segmentation losses
        seg_losses = self.seg_loss(outputs, targets)
        
        # Initialize total losses dictionary with segmentation losses
        losses = dict(seg_losses)
        
        # Compute temporal consistency loss if needed
        if flows is not None:
            temp_losses = self.temporal_loss(
                outputs.get('pred_masks', outputs.get('logits')), 
                flows
            )
            # Add temporal losses to the total losses
            for key, value in temp_losses.items():
                losses[key] = value
        
        # Compute total loss
        losses['total_loss'] = sum(loss for name, loss in losses.items() 
                                  if name != 'total_loss')
        
        return losses
```

# losses\segmentation.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,Tuple

# In losses/segmentation.py

# In losses/segmentation.py, modify the BinarySegmentationLoss class
class BinarySegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 1.5, boundary_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()  # New boundary loss
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract predictions and targets
        if 'logits' in outputs:
            logits = outputs['logits']  # [B, T, 1, H, W]
        else:
            # Ensure logits is defined by using pred_masks
            logits = outputs['pred_masks']  # May already be probabilities
            
        # Now logits is defined in all code paths
        ce_loss = self.ce_loss(logits.squeeze(1), binary_masks) * self.ce_weight
        dice_loss = self.dice_loss(pred_probs.squeeze(1), binary_masks) * self.dice_weight
        
        # Add boundary loss
        boundary_loss = self.boundary_loss(pred_probs.squeeze(1), binary_masks) * self.boundary_weight
        
        # Compute total loss
        total_loss = ce_loss + dice_loss + boundary_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss
        }

# Add a new boundary loss class
class BoundaryLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # Create boundary maps
        if self.sobel_x.device != predictions.device:
            self.sobel_x = self.sobel_x.to(predictions.device)
            self.sobel_y = self.sobel_y.to(predictions.device)
        
        # Get edges from predictions
        pred_boundaries = self._get_boundaries(predictions.unsqueeze(1))
        
        # Get edges from targets
        target_boundaries = self._get_boundaries(targets.unsqueeze(1).float())
        
        # Calculate boundary IoU loss
        intersection = (pred_boundaries * target_boundaries).sum(dim=[1, 2, 3])
        union = pred_boundaries.sum(dim=[1, 2, 3]) + target_boundaries.sum(dim=[1, 2, 3]) - intersection
        boundary_iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return (1 - boundary_iou).mean()
    
    def _get_boundaries(self, tensor):
        # Apply Sobel filters for edge detection
        grad_x = F.conv2d(tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor, self.sobel_y, padding=1)
        
        # Calculate gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and threshold
        grad_mag = grad_mag / grad_mag.max()
        return grad_mag
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss between predictions and targets.
        
        Args:
            predictions: Predicted probabilities [B, H, W]
            targets: Binary target masks [B, H, W]
            
        Returns:
            Dice loss
        """
        batch_size = predictions.size(0)
        
        # Flatten predictions and targets
        pred_flat = predictions.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Compute intersection and union
        intersection = (pred_flat * targets_flat).sum(1)
        union = pred_flat.sum(1) + targets_flat.sum(1)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice.mean()
```

# losses\temporal_consistency.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class TemporalConsistencyLoss(nn.Module):
    """Loss to enforce temporal consistency between frames."""
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        pred_masks: torch.Tensor,  # [B, T, C, H, W]
        flows: Optional[torch.Tensor] = None  # [B, T-1, 2, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate temporal consistency loss between consecutive frames.
        
        Args:
            pred_masks: Predicted segmentation masks
            flows: Optional optical flow between consecutive frames
            
        Returns:
            Dictionary containing:
                - 'temporal_loss': Main temporal consistency loss
                - 'smoothness_loss': Optional flow smoothness loss if flows provided
        """
        if pred_masks.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got shape {pred_masks.shape}")
            
        B, T, C, H, W = pred_masks.shape
        losses = {}
        
        # Basic temporal consistency - difference between consecutive frames
        temporal_diff = pred_masks[:, 1:] - pred_masks[:, :-1]  # [B, T-1, C, H, W]
        temporal_loss = F.mse_loss(temporal_diff, torch.zeros_like(temporal_diff))
        losses['temporal_loss'] = temporal_loss * self.consistency_weight
        
        # If flows provided, use them for warped consistency
        if flows is not None:
            warped_masks = []
            for t in range(T-1):
                curr_flow = flows[:, t]  # [B, 2, H, W]
                next_mask = pred_masks[:, t+1]  # [B, C, H, W]
                
                # Create sampling grid from flow
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=flows.device),
                    torch.arange(W, device=flows.device),
                    indexing='ij'
                )
                grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
                grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
                
                # Add flow to grid
                flow_grid = grid + curr_flow
                
                # Normalize grid coordinates to [-1, 1]
                flow_grid[:, 0] = 2.0 * flow_grid[:, 0] / (W - 1) - 1.0
                flow_grid[:, 1] = 2.0 * flow_grid[:, 1] / (H - 1) - 1.0
                
                # Reshape grid for grid_sample
                flow_grid = flow_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
                
                # Warp masks using flow
                warped_mask = F.grid_sample(
                    next_mask,
                    flow_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )
                warped_masks.append(warped_mask)
            
            warped_masks = torch.stack(warped_masks, dim=1)  # [B, T-1, C, H, W]
            
            # Calculate warped consistency loss
            warped_loss = F.mse_loss(
                pred_masks[:, :-1],
                warped_masks.detach()
            )
            losses['warped_loss'] = warped_loss * self.consistency_weight
            
            # Optional flow smoothness loss
            if self.training:
                flow_gradients_x = flows[:, :, :, :, 1:] - flows[:, :, :, :, :-1]
                flow_gradients_y = flows[:, :, :, 1:, :] - flows[:, :, :, :-1, :]
                smoothness_loss = (flow_gradients_x.abs().mean() + 
                                 flow_gradients_y.abs().mean())
                losses['smoothness_loss'] = smoothness_loss * 0.1
        
        return losses
```

# losses\video_instance_loss.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss."""
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss."""
        # Apply sigmoid if needed
        if not (0 <= pred.min() <= 1 and 0 <= pred.max() <= 1):
            pred = torch.sigmoid(pred)
        
        # For numerical stability
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        
        # Compute focal loss
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # Compute binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Combine
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Loss to encourage temporal consistency between consecutive frames.
    """
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        pred_masks: torch.Tensor,  # [B, T, N, H, W]
        flows: Optional[torch.Tensor] = None  # Optional flow fields
    ) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Simple temporal difference penalty
        if pred_masks.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,T,N,H,W], got shape {pred_masks.shape}")
            
        B, T, N, H, W = pred_masks.shape
        
        # No temporal loss for single frame
        if T <= 1:
            return torch.tensor(0.0, device=pred_masks.device)
        
        # Calculate difference between consecutive frames
        temporal_diff = pred_masks[:, 1:] - pred_masks[:, :-1]  # [B, T-1, N, H, W]
        
        # Compute L2 loss
        temporal_loss = F.mse_loss(temporal_diff, torch.zeros_like(temporal_diff))
        
        return temporal_loss * self.consistency_weight


class VideoInstanceSegmentationLoss(nn.Module):
    """
    Combined loss for video instance segmentation with temporal consistency.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        temporal_weight: float = 0.5
    ):
        super().__init__()
        
        # Segmentation losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
        # Temporal consistency loss
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
        
        # Loss weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        flows: Optional[torch.Tensor] = None  # Optional flow fields
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for video instance segmentation.
        
        Args:
            outputs: Dictionary containing 'pred_masks' [B, T, N, H, W]
            targets: Dictionary containing 'masks' [B, T, H, W]
            flows: Optional flow fields between frames
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Get predictions and ground truth
        pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
        gt_masks = targets['masks']         # [B, T, H, W]
        
        B, T, N, H, W = pred_masks.shape
        
        # Initialize losses
        dice_loss = 0
        focal_loss = 0
        
        # For each instance in prediction, find best matching ground truth
        instance_ids = torch.unique(gt_masks)[1:]  # Skip background
        
        # Simple matching strategy: for each ground truth instance, compute loss
        # against best matching predicted instance
        for instance_id in instance_ids:
            # Create binary mask for this instance
            gt_instance = (gt_masks == instance_id).float()  # [B, T, H, W]
            
            # Find best matching predicted instance
            best_iou = -1
            best_idx = -1
            
            for n in range(N):
                pred_instance = pred_masks[:, :, n]  # [B, T, H, W]
                
                # Compute IoU
                intersection = (pred_instance > 0.5).float() * gt_instance
                union = (pred_instance > 0.5).float() + gt_instance - intersection
                
                iou = (intersection.sum() + 1e-6) / (union.sum() + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = n
            
            if best_idx >= 0:
                # Compute losses for best matching instance
                pred_instance = pred_masks[:, :, best_idx]  # [B, T, H, W]
                
                # Dice loss
                dice = self.dice_loss(pred_instance, gt_instance)
                dice_loss += dice
                
                # Focal loss
                focal = self.focal_loss(pred_instance, gt_instance)
                focal_loss += focal
        
        # Normalize by number of instances
        num_instances = max(1, len(instance_ids))
        dice_loss /= num_instances
        focal_loss /= num_instances
        
        # Compute temporal consistency loss
        temp_loss = self.temporal_loss(pred_masks, flows)
        
        # Weighted combination
        total_loss = (
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss +
            temp_loss
        )
        
        return {
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'temporal_loss': temp_loss,
            'total_loss': total_loss
        }
```

# metrics\__init__.py

```py
from .evaluator import DAVISEvaluator

__all__ = [
    'DAVISEvaluator'
]
```

# metrics\contour.py

```py
import torch
import numpy as np
from typing import Dict, List

def compute_boundary_metrics(
    pred_masks: torch.Tensor,  # [T, C, H, W]
    gt_masks: torch.Tensor     # [T, H, W]
) -> Dict[str, float]:
    """
    Compute boundary-based F-measure metrics following DAVIS benchmark.
    
    Args:
        pred_masks: Predicted segmentation masks over time
        gt_masks: Ground truth masks over time
        
    Returns:
        Dictionary containing F-mean, F-recall, and F-decay
    """
    num_frames = pred_masks.shape[0]
    f_scores = []
    
    for t in range(num_frames):
        pred = pred_masks[t].argmax(0)
        gt = gt_masks[t]
        
        # Get boundaries
        pred_boundary = get_mask_boundary(pred)
        gt_boundary = get_mask_boundary(gt)
        
        # Compute precision and recall with tolerance
        precision = compute_boundary_precision(pred_boundary, gt_boundary)
        recall = compute_boundary_recall(pred_boundary, gt_boundary)
        
        # Compute F-measure
        f_score = 2 * precision * recall / (precision + recall + 1e-6)
        f_scores.append(f_score)
    
    f_mean = np.mean(f_scores)
    f_recall = np.mean([f > 0.5 for f in f_scores])
    f_decay = max(0, f_scores[0] - np.mean(f_scores[-4:]))
    
    return {
        'F_mean': f_mean,
        'F_recall': f_recall,
        'F_decay': f_decay
    }
```

# metrics\evaluator.py

```py
class DAVISEvaluator:
    """
    Main evaluator class that combines all DAVIS benchmark metrics.
    """
    def __init__(self):
        pass
    
    def evaluate_sequence(
        self,
        pred_masks: torch.Tensor,  # [T, C, H, W]
        gt_masks: torch.Tensor     # [T, H, W]
    ) -> Dict[str, float]:
        """
        Evaluate a single video sequence using all metrics.
        """
        metrics = {}
        
        # Region similarity (J)
        metrics.update(compute_region_metrics(pred_masks, gt_masks))
        
        # Boundary accuracy (F)
        metrics.update(compute_boundary_metrics(pred_masks, gt_masks))
        
        # Temporal stability (T)
        metrics.update(compute_temporal_metrics(pred_masks))
        
        return metrics
    
    def evaluate_dataset(
        self,
        predictions: List[torch.Tensor],
        ground_truths: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset and compute global metrics.
        """
        all_metrics = []
        
        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate_sequence(pred, gt)
            all_metrics.append(metrics)
        
        # Average across sequences
        global_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            global_metrics[key] = np.mean(values)
        
        return global_metrics
```

# metrics\region.py

```py
import torch
import numpy as np
from typing import Dict, List

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth masks.
    This is also known as the Jaccard index or J-score in DAVIS benchmark.
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
    
    Returns:
        IoU score between 0 and 1
    """
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return (intersection / (union + 1e-6)).item()

def compute_region_metrics(
    pred_masks: torch.Tensor,  # [T, C, H, W]
    gt_masks: torch.Tensor     # [T, H, W]
) -> Dict[str, float]:
    """
    Compute region-based metrics following DAVIS benchmark.
    
    Args:
        pred_masks: Predicted segmentation masks over time
        gt_masks: Ground truth masks over time
    
    Returns:
        Dictionary containing J-mean, J-recall, and J-decay
    """
    num_frames = pred_masks.shape[0]
    ious = []
    
    # Compute IoU for each frame
    for t in range(num_frames):
        pred = pred_masks[t].argmax(0)  # Convert to class indices
        gt = gt_masks[t]
        ious.append(compute_iou(pred == 1, gt == 1))
    
    # Compute metrics
    j_mean = np.mean(ious)
    j_recall = np.mean([iou > 0.5 for iou in ious])  # % of frames with IoU > 0.5
    
    # Compute decay (difference between first and last frames)
    j_decay = max(0, ious[0] - np.mean(ious[-4:]))  # Following DAVIS protocol
    
    return {
        'J_mean': j_mean,
        'J_recall': j_recall,
        'J_decay': j_decay
    }

```

# metrics\temporal.py

```py
import torch
import numpy as np
from typing import Dict, List


def compute_temporal_metrics(
    pred_masks: torch.Tensor,  # [T, C, H, W]
) -> Dict[str, float]:
    """
    Compute temporal stability metrics following DAVIS benchmark.
    
    Args:
        pred_masks: Predicted segmentation masks over time
        
    Returns:
        Dictionary containing temporal stability metrics
    """
    num_frames = pred_masks.shape[0]
    
    # Convert to binary masks
    pred_masks = pred_masks.argmax(1)  # [T, H, W]
    
    # Compute frame-to-frame changes
    changes = []
    for t in range(num_frames - 1):
        change = (pred_masks[t+1] != pred_masks[t]).float().mean()
        changes.append(change.item())
    
    # Lower values indicate better stability
    t_mean = 1.0 - np.mean(changes)
    
    return {'T_mean': t_mean}

```

# models\__init__.py

```py
# models/__init__.py

import os
import sys
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Add Mask2Former to Python path
mask2former_path = PROJECT_ROOT / 'Mask2Former'
if mask2former_path.exists():
    if str(mask2former_path) not in sys.path:
        sys.path.append(str(mask2former_path))
    print(f"Using Mask2Former from: {mask2former_path}")
else:
    print(f"Note: Mask2Former not found at {mask2former_path}, using custom video decoder instead.")

from .backbone import BackboneEncoder, CNNBackbone, VideoMambaBlock, TemporalFeatureBank
from .video_instance_decoder import VideoInstanceDecoder, InstanceMemory
from .temporal_components import EnhancedTemporalSmoothingModule
from .video_model import VideoMambaSegmentation, build_model

__all__ = [
    'BackboneEncoder',
    'CNNBackbone',
    'VideoMambaBlock',
    'TemporalFeatureBank',
    'VideoInstanceDecoder',
    'InstanceMemory',
    'EnhancedTemporalSmoothingModule',
    'VideoMambaSegmentation',
    'build_model'
]
```

# models\backbone.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba as MambaBlock
from collections import deque
from typing import Optional, Tuple, List

class CNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Conv2d(curr_dim, hidden_dim, 3, padding=1),
                # Changed LayerNorm to BatchNorm2d for 4D tensors
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
            curr_dim = hidden_dim
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            features.append(current)
            
        return features

class VideoMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize Mamba block
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand = expand
        )
        
        self.state_reset_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.temporal_state = None
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape spatial dimensions into sequence length
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
    
    def _restore_spatial(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # Restore spatial dimensions
        b, hw, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x
        
    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False,
        motion_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not x.is_cuda:
            x = x.cuda()
        b, c, h, w = x.shape
        

        # Prepare input for Mamba
        x_seq = self._prepare_input(x)  # [B, H*W, C]
        
        # Apply dropout
        x_seq = self.dropout(x_seq)
        
        # Process through Mamba
        x_seq = self.mamba(x_seq)
        
        # Restore spatial dimensions
        x = self._restore_spatial(x_seq, h, w)
        
        # For now, just return processed tensor and None for state
        return x, None

class TemporalFeatureBank(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        window_size: int = 5,
        confidence_threshold: float = 0.7
    ):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.confidence_threshold = confidence_threshold
        self.features = deque(maxlen=window_size)
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            # Changed LayerNorm to BatchNorm2d
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def update(
        self,
        current_features: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> None:
        mask = confidence_scores > self.confidence_threshold
        self.features.append((current_features.detach(), mask))
        
    def get_temporal_context(self, current_features: torch.Tensor) -> torch.Tensor:
        if not self.features:
            return current_features
            
        # Get current feature dimensions
        B, C, H, W = current_features.shape
        
        temporal_features = []
        for hist_features, mask in self.features:
            # Skip features with wrong batch size
            if hist_features.shape[0] != B:
                continue
                
            # Resize historical features to match current feature size if needed
            if hist_features.shape[2:] != current_features.shape[2:]:
                hist_features = F.interpolate(
                    hist_features, 
                    size=(H, W),
                    mode='bilinear', 
                    align_corners=False
                )
                # Also resize mask
                if mask.shape[2:] != current_features.shape[2:]:
                    mask_float = mask.float()
                    mask_float = F.interpolate(
                        mask_float, 
                        size=(H, W),
                        mode='nearest'
                    )
                    mask = mask_float > 0.5
                    
            temporal_features.append(hist_features * mask)
            
        # Safely compute mean with proper reshaping
        if temporal_features:
            temporal_context = torch.cat([
                current_features,
                torch.mean(torch.stack(temporal_features), dim=0)
            ], dim=1)
        else:
            # If no temporal features yet, duplicate current features
            temporal_context = torch.cat([current_features, current_features], dim=1)
        
        return self.feature_fusion(temporal_context)

class BackboneEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        d_state: int = 16,
        temporal_window: int = 5,
        dropout: float = 0.1,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        self.cnn_backbone = CNNBackbone(input_dim, hidden_dims)
        
        self.mamba_blocks = nn.ModuleList([
            VideoMambaBlock(
                d_model=dim,
                d_state=d_state,
                d_conv = d_conv,
                expand = expand,
                dropout=dropout
            )
            for dim in hidden_dims
        ])
        
        self.temporal_banks = nn.ModuleList([
            TemporalFeatureBank(
                feature_dim=dim,
                window_size=temporal_window
            )
            for dim in hidden_dims
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        reset_states: bool = False,
        motion_info: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)
        
        # Process through Mamba blocks and Temporal Feature Banks
        enhanced_features = []
        for feat, mamba, bank in zip(cnn_features, self.mamba_blocks, self.temporal_banks):
            # Mamba processing
            mamba_features, _ = mamba(feat, reset_states, motion_info)
            
            # Temporal bank processing
            confidence = torch.norm(mamba_features, dim=1, keepdim=True)
            bank.update(mamba_features, confidence)
            temporal_features = bank.get_temporal_context(mamba_features)
            
            enhanced_features.append(temporal_features)
            
        return enhanced_features
```

# models\decoder.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from typing import List, Optional, Dict, Tuple, Union
from .temporal_components import FlexibleTemporalAttention, EnhancedTemporalSmoothingModule


class MaskFeatureProjection(nn.Module):
    """
    Projects features with progressive channel scaling to match Mask2Former's expectations.
    Features are projected up to higher dimensions before being processed by the decoder.
    """
    def __init__(self, in_channels: Union[List[int], int], mask_dim: int):
        super().__init__()
        self.mask_dim = mask_dim
        
        if isinstance(in_channels, list):
            # For backbone features: project each level up to target dimensions
            # We project to higher dimensions first, then let the decoder reduce them
            target_dims = [mask_dim // 2, mask_dim // 2, mask_dim]  # Progressive scaling
            
            self.projections = nn.ModuleList([
                nn.Sequential(
                    # First increase channels to intermediate dimension
                    nn.Conv2d(in_chan, target_dim, 3, padding=1),
                    nn.GroupNorm(8, target_dim),
                    nn.ReLU(inplace=True),
                    # Then refine features
                    nn.Conv2d(target_dim, target_dim, 1),
                    nn.GroupNorm(8, target_dim),
                    nn.ReLU(inplace=True)
                ) for in_chan, target_dim in zip(in_channels, target_dims)
            ])
            self.is_backbone = True
            
            print("\nFeature projection dimensions:")
            for in_chan, target_dim in zip(in_channels, target_dims):
                print(f"- Level {len(self.projections)}: {in_chan} -> {target_dim} channels")
        else:
            # For mask features: project to mask dimension
            self.projections = nn.Sequential(
                nn.Conv2d(in_channels, mask_dim, 3, padding=1),
                nn.GroupNorm(8, mask_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(mask_dim, mask_dim, 1),
                nn.GroupNorm(8, mask_dim),
                nn.ReLU(inplace=True)
            )
            self.is_backbone = False
    
    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Projects features with detailed shape tracking for easier debugging.
        
        Args:
            x: Either a list of backbone features or a single mask feature tensor
        Returns:
            Projected features with appropriate dimensions
        """
        if self.is_backbone:
            projected = []
            for i, (feat, proj) in enumerate(zip(x, self.projections)):
                # Print shapes for debugging
                print(f"Level {i + 1} projection: {feat.shape} -> ", end='')
                out = proj(feat)
                print(f"{out.shape}")
                projected.append(out)
            return projected
        else:
            # Process mask features
            if x.dim() == 5:  # [B, T, C, H, W]
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
            
            # Project to mask dimension
            projected = self.projections(x)
            print(f"Mask feature projection: {x.shape} -> {projected.shape}")
            return projected

from typing import List, Optional, Dict, Tuple, Union



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .position_encoding import LeveledPositionEmbeddingSine

class MambaMask2FormerDecoder(nn.Module):
    """
    Enhanced decoder that maintains consistent dimensions throughout the pipeline
    and processes frames individually to avoid dimension conflicts.
    """
    def __init__(
        self,
        in_channels: List[int],
        mask2former_config: dict,
        num_classes: int,
        mask_dim: int = 256
    ):
        super().__init__()
        self.mask_dim = mask_dim
        
        # We'll project everything to mask_dim for consistency
        self.target_dim = mask_dim
        
        print(f"\nInitializing MambaMask2FormerDecoder:")
        print(f"- Input channels: {in_channels}")
        print(f"- Target dimension: {self.target_dim}")
        print(f"- Number of classes: {num_classes}")
        
        # Feature projection - now all levels go directly to target_dim
        self.input_projections = nn.ModuleList([
            nn.Sequential(
                # First projection handles dimension change
                nn.Conv2d(in_dim, self.target_dim, 3, padding=1),
                nn.GroupNorm(8, self.target_dim),
                nn.ReLU(inplace=True),
                # Second convolution refines features
                nn.Conv2d(self.target_dim, self.target_dim, 1),
                nn.GroupNorm(8, self.target_dim),
                nn.ReLU(inplace=True)
            ) for in_dim in in_channels
        ])
        
        # Position encodings match final dimension
        self.pos_encodings = nn.ModuleList([
            LeveledPositionEmbeddingSine(self.target_dim)
            for _ in in_channels
        ])
        
        # Mask feature projection to match final dimension
        self.mask_feature_projection = nn.Sequential(
            nn.Conv2d(mask_dim, self.target_dim, 3, padding=1),
            nn.GroupNorm(8, self.target_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.target_dim, self.target_dim, 1),
            nn.GroupNorm(8, self.target_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize the original Mask2Former decoder
        self.original_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=self.target_dim,
            num_classes=num_classes,
            hidden_dim=mask2former_config['hidden_dim'],
            num_queries=mask2former_config['num_queries'],
            nheads=mask2former_config['nheads'],
            dim_feedforward=mask2former_config['dim_feedforward'],
            dec_layers=mask2former_config['dec_layers'],
            pre_norm=True,
            mask_dim=self.target_dim,
            enforce_input_project=False
        )
    
    def _prepare_single_frame(
        self,
        features: List[torch.Tensor],
        mask_features: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares features for a single frame.
        
        Args:
            features: List of feature tensors for a single frame [1, C, H, W]
            mask_features: Optional mask features for a single frame [1, C, H, W]
            
        Returns:
            Tuple of (projected_features, position_encodings, prepared_mask_features)
        """
        # Project features
        projected_features = []
        for i, (feat, proj) in enumerate(zip(features, self.input_projections)):
            projected_features.append(proj(feat))
        
        # Generate position encodings
        pos_encodings = []
        for i, (feat, pos_enc) in enumerate(zip(projected_features, self.pos_encodings)):
            encoding = pos_enc(feat)
            pos_encodings.append(encoding)
        
        # Process mask features if provided
        prepared_mask_features = None
        if mask_features is not None:
            prepared_mask_features = self.mask_feature_projection(mask_features)
        
        return projected_features, pos_encodings, prepared_mask_features
    
    def _process_single_frame(
        self, 
        features: List[torch.Tensor],
        mask_features: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single frame through the Mask2Former decoder.
        
        Args:
            features: List of feature tensors for a single frame [1, C, H, W]
            mask_features: Optional mask features for a single frame [1, C, H, W]
            targets: Optional target dictionaries for training
            
        Returns:
            Dictionary of outputs for this frame
        """
        # Prepare features for this frame
        proj_features, pos_encodings, prepared_mask = self._prepare_single_frame(
            features, mask_features
        )
        
        # Process through original decoder
        if self.training and targets is not None:
            outputs = self.original_decoder(
                proj_features,
                prepared_mask,
                pos_encodings,
                targets
            )
        else:
            outputs = self.original_decoder(
                proj_features,
                prepared_mask,
                pos_encodings
            )
        
        return outputs
    
    def _validate_dimensions(
        self,
        features: List[torch.Tensor],
        pos_encodings: List[torch.Tensor],
        mask_features: Optional[torch.Tensor]
    ):
        """Validates that all dimensions match the target dimension."""
        for i, feat in enumerate(features):
            assert feat.shape[1] == self.target_dim, \
                f"Feature level {i} has {feat.shape[1]} channels, expected {self.target_dim}"
            assert pos_encodings[i].shape[1] == self.target_dim, \
                f"Position encoding level {i} has {pos_encodings[i].shape[1]} channels, expected {self.target_dim}"
            
        if mask_features is not None:
            assert mask_features.shape[1] == self.target_dim, \
                f"Mask features have {mask_features.shape[1]} channels, expected {self.target_dim}"
    
    def forward(
        self,
        features: List[torch.Tensor],
        mask_features: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder with frame-by-frame processing to avoid dimension conflicts.
        
        Args:
            features: List of backbone feature tensors [B*T, C, H, W]
            mask_features: Optional mask features tensor [B*T, C, H, W] or [B, T, C, H, W]
            targets: Optional target dictionaries for training
            
        Returns:
            Dictionary containing model outputs (pred_masks, aux_outputs, etc.)
        """
        try:
            print("\nFeature preparation pipeline:")
            print(f"1. Input backbone features: {[f.shape for f in features]}")
            
            # Detect if we have temporal dimension in mask_features
            has_temporal_dim = mask_features is not None and mask_features.dim() == 5
            
            # Get batch size and time steps
            if has_temporal_dim:
                B, T = mask_features.shape[:2]
                # Flatten mask features for consistent processing
                mask_features = mask_features.reshape(B * T, *mask_features.shape[2:])
            else:
                # Assume we've already flattened time dimension into batch
                # Infer from the first feature's batch dimension
                batch_dim = features[0].shape[0]
                B = 1  # Default single video processing
                T = batch_dim // B  # Number of frames
            
            # Process each frame individually to avoid dimension conflicts
            frame_outputs = []
            
            for t in range(T):
                # Extract features for this frame
                frame_features = [feat[t:t+1] for feat in features]
                
                # Extract mask features for this frame if available
                frame_mask = None
                if mask_features is not None:
                    frame_mask = mask_features[t:t+1]
                
                # Process through decoder
                output = self._process_single_frame(
                    frame_features, frame_mask, targets
                )
                
                frame_outputs.append(output)
            
            # Combine outputs from all frames
            combined_output = {}
            
            # Handle 'pred_masks' - stack along batch dimension
            if 'pred_masks' in frame_outputs[0]:
                pred_masks = torch.cat([out['pred_masks'] for out in frame_outputs], dim=0)
                combined_output['pred_masks'] = pred_masks
            
            # Copy other output keys from the last frame
            for key in frame_outputs[-1].keys():
                if key != 'pred_masks':
                    combined_output[key] = frame_outputs[-1][key]
            
            return combined_output
            
        except Exception as e:
            print("\nError in decoder forward pass:")
            print(f"Feature shapes: {[f.shape for f in features]}")
            if mask_features is not None:
                print(f"Mask features shape: {mask_features.shape}")
            raise e


class TemporalFeatureAdapter(nn.Module):
    """
    Adapts features to maintain consistency across video frames.
    Uses temporal convolution to model relationships across time.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Temporal mixing with 3D convolution
        self.temporal_conv = nn.Conv3d(
            feature_dim, feature_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=8
        )
        
        # Feature refinement
        self.feature_norm = nn.GroupNorm(8, feature_dim)
        self.feature_act = nn.ReLU(inplace=True)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal adaptation to features.
        
        Args:
            features: Input features with temporal dimension [B, T, C, H, W]
            
        Returns:
            Temporally adapted features [B, T, C, H, W]
        """
        # Apply temporal convolution
        identity = features
        features = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        features = self.temporal_conv(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)
        features = features.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        
        # Residual connection
        return features + identity

        
class SegmentationHead(nn.Module):
    """
    Enhanced segmentation head that handles both instance segmentation and temporal consistency.
    Works with the updated MambaMask2FormerDecoder for proper temporal handling.
    """
    def __init__(
        self,
        in_channels: List[int],
        mask2former_config: dict,
        num_classes: int = 1  # Single class for instance segmentation
    ):
        super().__init__()
        self.mask_dim = mask2former_config.get('mask_dim', 256)
        
        # Initialize decoder with frame-by-frame processing capability
        self.decoder = MambaMask2FormerDecoder(
            in_channels=in_channels,
            mask2former_config=mask2former_config,
            num_classes=num_classes,
            mask_dim=self.mask_dim
        )
        
        # Initialize temporal smoothing
        self.temporal_smooth = EnhancedTemporalSmoothingModule(
            channels=mask2former_config['num_queries']
        )

    def _process_per_frame(
        self,
        features: List[torch.Tensor],
        mask_features: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process each frame individually to avoid temporal dimension conflicts.
        
        This is the key function that solves the dimension mismatch by:
        1. Detecting the number of frames
        2. Processing each frame separately
        3. Combining the results afterward
        """
        # Get batch size and number of frames
        B = 1  # Since we're processing one video at a time
        T = features[0].shape[0] // B  # Number of frames
        
        # Create storage for results
        all_pred_masks = []
        
        # Process each frame separately
        for t in range(T):
            # Extract features for this frame
            frame_features = [feat[t:t+1] for feat in features]
            
            # Extract mask features for this frame if available
            frame_mask = None
            if mask_features is not None:
                frame_mask = mask_features[t:t+1]
            
            # Project features for each level
            projected_features = []
            pos_encodings = []
            
            for i, (feat, proj, pos_enc) in enumerate(zip(frame_features, self.feature_projections, self.position_encodings)):
                # Project features
                proj_feat = proj(feat)
                projected_features.append(proj_feat)
                
                # Generate position encoding
                pos = pos_enc(proj_feat)
                pos_encodings.append(pos)
            
            # Process mask features if available
            if frame_mask is not None:
                frame_mask = self.mask_projection(frame_mask)
            
            # Process through Mask2Former decoder
            # Now this works because we're only passing one frame at a time
            if self.training and targets is not None:
                # Adjust targets for this frame if needed
                frame_targets = targets  # You may need to modify this based on your target format
                outputs = self.decoder(
                    projected_features,
                    frame_mask,
                    pos_encodings,
                    frame_targets
                )
            else:
                outputs = self.decoder(
                    projected_features,
                    frame_mask,
                    pos_encodings
                )
            
            # Store masks for this frame
            all_pred_masks.append(outputs['pred_masks'])
        
        # Combine results from all frames
        combined_masks = torch.cat(all_pred_masks, dim=0)
        
        # Create final output dictionary
        results = {'pred_masks': combined_masks}
        
        # Add any other outputs needed
        for k, v in outputs.items():
            if k != 'pred_masks':
                results[k] = v
        
        return results
        
    def forward(
        self,
        features: List[torch.Tensor],
        mask_features: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the segmentation head with proper temporal handling.
        
        Args:
            features: List of backbone feature tensors
            mask_features: Optional mask features tensor
            targets: Optional target dictionaries for training
        """
        # Process features through decoder - the decoder now handles frame-by-frame processing
        if self.training and targets is not None:
            outputs = self.decoder(features, mask_features, targets)
        else:
            outputs = self.decoder(features, mask_features)
        
        # If we have temporal dimension in mask_features
        if 'pred_masks' in outputs and mask_features is not None:
            pred_masks = outputs['pred_masks']
            
            # Check if mask_features has temporal dimension
            has_temporal_dim = mask_features.dim() == 5
            
            if has_temporal_dim:
                B, T = mask_features.shape[:2]
                H, W = pred_masks.shape[-2:]
                N = pred_masks.shape[1]  # Number of instances
                
                # Reshape for temporal smoothing
                pred_masks = pred_masks.view(B, T, N, H, W)
                
                # Apply temporal smoothing
                smoothed_masks = self.temporal_smooth(pred_masks)
                
                # Prepare final output
                outputs['pred_masks'] = smoothed_masks.flatten(0, 1)
            
        return outputs
```

# models\mask2former_integration.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher


class Mask2FormerIntegration(nn.Module):
    """Integrates backbone features with Mask2Former"""
    def __init__(
        self,
        in_channels: List[int],
        mask2former_config: Dict,
        num_classes: int,
        mask_dim: int = 256,
    ):
        super().__init__()
        
        # Feature projection for masks
        self.mask_projection = MaskFeatureProjection(in_channels, mask_dim)
        
        # Position encoding
        self.pos_encoding = PositionEmbeddingSine(
            mask2former_config['hidden_dim'] // 2
        )
        
        # Extract decoder-specific parameters
        decoder_params = {
            'hidden_dim': mask2former_config['hidden_dim'],
            'num_queries': mask2former_config['num_queries'],
            'nheads': mask2former_config['nheads'],
            'dim_feedforward': mask2former_config.get('dim_feedforward', 2048),
            'dec_layers': mask2former_config.get('dec_layers', 9),
            'pre_norm': True,
            'enforce_input_project': mask2former_config.get('enforce_input_project', False),
            'mask_dim': mask_dim
        }
        
        # Initialize Mask2Former decoder with filtered parameters
        self.decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=mask_dim,
            num_classes=num_classes,
            **decoder_params
        )
        
        # Initialize matcher
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0
        )
        
        weight_dict = {
            "loss_ce": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0,
        }
        
        # Extract criterion parameters
        criterion_params = {
            'num_points': mask2former_config.get('num_points', 12544),
            'oversample_ratio': mask2former_config.get('oversample_ratio', 3.0),
            'importance_sample_ratio': mask2former_config.get('importance_sample_ratio', 0.75)
        }
        
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "masks"],
            **criterion_params
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        # Project features and get pos encodings
        projected_features = self.mask_projection(features)
        pos_encodings = [self.pos_encoding(feat) for feat in projected_features]
        
        if self.training and targets is not None:
            # Training mode
            outputs = self.decoder(projected_features, None, pos_encodings, targets)
            losses = self.criterion(outputs, targets)
            return outputs, losses
        else:
            # Inference mode
            outputs = self.decoder(projected_features, None, pos_encodings)
            return outputs, None

def build_mask2former_integration(config):
    """Builds the Mask2Former integration module from config"""
    return Mask2FormerIntegration(
        in_channels=config.in_channels,
        mask2former_config=config.mask2former,
        num_classes=config.num_classes,
        mask_dim=config.mask2former.get('mask_dim', 256)
    )
```

# models\model.py

```py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .backbone import BackboneEncoder
from .decoder import MambaMask2FormerDecoder
from .temporal_components import (
    InstanceMotionModule,
    InstanceTemporalAttention,
    EnhancedTemporalSmoothingModule
)

class VideoMambaSegmentation(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.feature_dim = config['hidden_dims'][-1]
        self.mask_dim = config['mask2former']['mask_dim']
        self.num_instances = config['mask2former']['num_queries']
        
        print(f"\nInitializing complete video instance segmentation model:")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Mask dimension: {self.mask_dim}")
        print(f"- Number of instances: {self.num_instances}")
        
        # Initialize backbone
        self.backbone = BackboneEncoder(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            d_state=config['d_state'],
            temporal_window=config['temporal_window'],
            dropout=config.get('dropout', 0.1)
        )
        
        # Initialize instance-specific temporal components
        self.instance_motion = InstanceMotionModule(
            feature_dim=self.feature_dim,
            num_instances=self.num_instances
        )
        
        self.temporal_attention = InstanceTemporalAttention(
            feature_dim=self.feature_dim,
            num_instances=self.num_instances,
            num_heads=8
        )
        
        # Initialize mask projection
        self.mask_projection = nn.Sequential(
            nn.Conv2d(config['input_dim'], self.mask_dim // 2, 3, padding=1),
            nn.BatchNorm2d(self.mask_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mask_dim // 2, self.mask_dim, 1),
            nn.BatchNorm2d(self.mask_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize decoder with proper mask dimension
        self.decoder = MambaMask2FormerDecoder(
            in_channels=config['hidden_dims'],
            mask2former_config=config['mask2former'],
            num_classes=1  # Binary segmentation per instance
        )
        
        # Initialize temporal smoothing
        self.temporal_smooth = EnhancedTemporalSmoothingModule(
            channels=self.num_instances,
            temporal_kernel=3
        )
    
    def _generate_mask_features(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Generate mask features with temporal awareness."""
        # Project to mask dimension
        mask_features = self.mask_projection(x)  # [B*T, mask_dim, H, W]
        _, C, H, W = mask_features.shape
        return mask_features.view(B, T, C, H, W)
    
    def _process_temporal_features(
        self,
        features: torch.Tensor,
        B: int,
        T: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process features through temporal components."""
        C, H, W = features.shape[1:]
        features = features.view(B, T, C, H, W)
        
        # Track instance motion
        motion_features, motion_field = self.instance_motion(features)
        
        # Apply instance-specific temporal attention
        temporal_features = self.temporal_attention(
            motion_features,
            motion_field=motion_field
        )
        
        # Reshape back to batch format
        processed_features = temporal_features.reshape(B * T, C, H, W)
        return processed_features, motion_field
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with proper dimension handling for video data.
        
        Args:
            x: Input tensor [B, T, C, H, W] representing a video sequence
                
        Returns:
            Dictionary containing model outputs
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Generate mask features
        mask_features = self._generate_mask_features(x, B, T)
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Process through temporal components
        enhanced_features, motion_field = self._process_temporal_features(
            backbone_features[-1], B, T
        )
        backbone_features[-1] = enhanced_features
        
        # Process through decoder - no need to handle frames separately as it's done inside the decoder
        outputs = self.decoder(
            features=backbone_features,
            mask_features=mask_features
        )
        
        # Add motion field to outputs if available
        if motion_field is not None:
            outputs['motion_field'] = motion_field
        
        return outputs
    
    
def build_model(config: Dict) -> VideoMambaSegmentation:
    """
    Builds and initializes the model from config.
    Handles both nested and flat configuration formats.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
        
    # Handle nested configuration
    model_config = config.get('model', config)
    
    # Verify required parameters
    required_params = ['input_dim', 'hidden_dims', 'mask2former']
    missing_params = [param for param in required_params if param not in model_config]
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Create model
    return VideoMambaSegmentation(config)
```

# models\position_encoding.py

```py
import torch
import torch.nn as nn
import math

class LeveledPositionEmbeddingSine(nn.Module):
    """
    Position encoding that exactly matches feature dimensions at each level.
    The encoding dimension will precisely match the input feature dimension,
    ensuring compatibility throughout the feature hierarchy.
    """
    def __init__(self, feature_dim: int, temperature: int = 10000):
        super().__init__()
        # The feature dimension determines the number of position encoding channels
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Number of position encoding features matches input exactly
        self.num_pos_feats = feature_dim
        
        print(f"Initializing position embedding with {feature_dim} channels")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate position encodings that exactly match input dimensions.
        
        Args:
            x: Input tensor [B, C, H, W] where C is feature_dim
            
        Returns:
            Position encodings [B, C, H, W] with same channel count as input
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape: {x.shape}")
        
        if x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Input has {x.shape[1]} channels but position encoding "
                f"was initialized for {self.feature_dim} channels"
            )
        
        # Generate normalized coordinate grids
        not_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        
        # Create position encoding with correct number of channels
        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Stack sin and cos embeddings
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        
        # Combine x and y embeddings to get final position encoding
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Double check output dimensions
        assert pos.shape == x.shape, (
            f"Position encoding shape {pos.shape} doesn't match input shape {x.shape}"
        )
        
        return pos
```

# models\temporal_components.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class DimensionAdapter(nn.Module):
    """
    Handles dimension adaptation between different parts of the model.
    This module ensures smooth transitions between different feature spaces,
    acting like a universal translator between feature dimensions.
    """
    def __init__(self, feature_dim: int, mask_dim: int):
        super().__init__()
        
        # Feature space transformations
        self.feature_to_mask = nn.Sequential(
            nn.Linear(feature_dim, mask_dim),
            nn.LayerNorm(mask_dim),
            nn.ReLU(inplace=True)
        )
        
        self.mask_to_feature = nn.Sequential(
            nn.Linear(mask_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def adapt_features(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Adapts feature dimensions based on target size.
        Args:
            x: Input tensor
            target_dim: Desired output dimension
        """
        curr_dim = x.shape[-1]
        if curr_dim == target_dim:
            return x
        elif target_dim == self.mask_to_feature[-2].normalized_shape[0]:
            return self.mask_to_feature(x)
        else:
            return self.feature_to_mask(x)

class FlexibleTemporalAttention(nn.Module):
    """
    Temporal attention module that automatically handles dimension matching
    between features and queries.
    """
    def __init__(self, feature_dim: int, mask_dim: int, num_instances: int):
        super().__init__()
        
        self.dim_adapter = DimensionAdapter(feature_dim, mask_dim)
        
        # Temporal processing
        self.temporal_conv = nn.Conv3d(
            feature_dim, feature_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=feature_dim // 16 if feature_dim >= 16 else 1
        )
        
        # Cross-attention for instance features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim * 2, feature_dim, 1),
            nn.GroupNorm(min(8, feature_dim), feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, feature_dim, 32, 1, 1))
    
    def forward(self, features: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Process features with automatic dimension handling.
        Args:
            features: [B, C, T, H, W] backbone features
            queries: [B, N, D] instance queries
        """
        B, C, T, H, W = features.shape
        
        # Adapt query dimensions to match feature space
        adapted_queries = self.dim_adapter.adapt_features(queries, C)
        
        # Reshape features for attention
        feat_flat = features.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        feat_flat = feat_flat.reshape(B, T*H*W, C)   # [B, THW, C]
        
        # Apply cross-attention
        attended_feats, _ = self.cross_attention(
            feat_flat, adapted_queries, adapted_queries
        )
        
        # Reshape back to feature format
        attended_feats = attended_feats.view(B, T, H, W, C)
        attended_feats = attended_feats.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        
        # Apply temporal processing
        features = features + self.pos_encoding[:, :, :T]
        temporal_feats = self.temporal_conv(features)
        
        # Combine features
        combined = torch.cat([attended_feats, temporal_feats], dim=1)
        enhanced = self.feature_refine(combined)
        
        return enhanced

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class InstanceMotionModule(nn.Module):
    """
    Module responsible for tracking instance-specific motion between frames.
    This helps maintain instance identity by understanding how objects move
    through the video sequence.
    """
    def __init__(self, feature_dim: int, num_instances: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_instances = num_instances
        
        # Motion estimation network
        self.motion_estimator = nn.Sequential(
            # First layer processes features
            nn.Conv3d(feature_dim, feature_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True),
            # Second layer refines motion features
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True),
            # Final layer predicts motion field
            nn.Conv3d(feature_dim, 2, kernel_size=1)  # 2 channels for x,y motion
        )
        
        # Feature refinement with motion awareness
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim + 2, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process features to estimate motion and enhance feature representation.
        
        Args:
            features: Input features [B, T, C, H, W]
            
        Returns:
            Tuple containing:
            - Enhanced features incorporating motion information
            - Estimated motion field between consecutive frames
        """
        B, T, C, H, W = features.shape
        
        # Reshape for 3D convolution
        x = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Estimate motion
        motion = self.motion_estimator(x)  # [B, 2, T, H, W]
        
        # Get motion field between consecutive frames
        motion_field = motion.permute(0, 2, 1, 3, 4)  # [B, T, 2, H, W]
        
        # Combine features with motion information
        motion_features = torch.cat([x, motion], dim=1)
        enhanced = self.feature_refine(motion_features)
        
        # Return to original format
        enhanced = enhanced.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        
        return enhanced, motion_field

class InstanceTemporalAttention(nn.Module):
    """
    Enhanced temporal attention module that maintains instance identity across video frames
    while properly handling the output shapes to ensure compatibility with the spatial dimensions.
    """
    def __init__(
        self,
        feature_dim: int,
        num_instances: int,
        num_heads: int = 8
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_instances = num_instances
        self.num_heads = num_heads
        
        # Initialize learnable instance query embeddings
        self.instance_queries = nn.Parameter(
            torch.randn(num_instances, feature_dim)
        )
        
        # Multi-head attention for instance-feature interactions
        self.instance_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Temporal processing with 3D convolution
        self.temporal_conv = nn.Conv3d(
            feature_dim,
            feature_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=feature_dim // 16 if feature_dim >= 16 else 1
        )
        
        # Feature refinement combining temporal and instance information
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim * 2, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Position encoding for temporal awareness
        self.pos_encoding = nn.Parameter(
            torch.randn(1, feature_dim, 32, 1, 1)
        )
        
        # Instance feature projection to spatial domain
        self.instance_to_spatial = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        motion_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process features with both temporal attention and instance tracking,
        with corrected reshaping to maintain spatial dimensions.
        
        Args:
            features: Input features [B, T, C, H, W]
            motion_field: Optional motion information [B, T-1, 2, H, W]
            
        Returns:
            Enhanced features with temporal and instance awareness [B, T, C, H, W]
        """
        B, T, C, H, W = features.shape
        
        # Print shapes for debugging
        # print(f"\nProcessing in temporal attention:")
        # print(f"Input features shape: {features.shape}")
        
        # 1. General temporal processing with 3D convolution
        temporal_feats = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        temporal_feats = temporal_feats + self.pos_encoding[:, :, :T]
        temporal_feats = self.temporal_conv(temporal_feats)
        # print(f"After temporal conv: {temporal_feats.shape}")
        
        # 2. Instance-specific processing - first get feature vectors
        # Reshape features for attention by flattening spatial dimensions
        feat_flat = features.reshape(B * T, H * W, C)
        # print(f"Flattened features: {feat_flat.shape}")
        
        # Reshape instance queries for attention
        queries = self.instance_queries.unsqueeze(0).expand(B * T, -1, -1)
        # print(f"Instance queries shape: {queries.shape}")
        
        # Apply instance-specific attention
        # Instance queries attend to spatial features
        instance_feats, _ = self.instance_attention(
            queries,            # Queries from instance embeddings [B*T, N, C]
            feat_flat,          # Keys from flattened features [B*T, H*W, C]
            feat_flat           # Values from flattened features [B*T, H*W, C]
        )
        # print(f"After attention: {instance_feats.shape}")  # [B*T, N, C]
        
        # 3. Project instance features back to spatial domain
        # We use a different approach that doesn't require reshape to full spatial size
        
        # First get instance features for each position
        # Create attention map from instances to spatial locations
        instance_attn = torch.bmm(
            feat_flat,                          # [B*T, H*W, C]
            instance_feats.transpose(1, 2)      # [B*T, C, N]
        )  # [B*T, H*W, N]
        
        # Normalize attention weights
        instance_attn = F.softmax(instance_attn, dim=2)
        
        # Get weighted instance features for each spatial location
        spatial_instance_feats = torch.bmm(
            instance_attn,                      # [B*T, H*W, N]
            instance_feats                      # [B*T, N, C]
        )  # [B*T, H*W, C]
        
        # Reshape back to spatial format
        spatial_instance_feats = spatial_instance_feats.reshape(B, T, H, W, C)
        spatial_instance_feats = spatial_instance_feats.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        
        # Now reshape to match the temporal processing format
        instance_spatial = spatial_instance_feats.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        # print(f"Instance features mapped to spatial: {instance_spatial.shape}")
        
        # 4. Combine temporal and instance features
        combined = torch.cat([temporal_feats, instance_spatial], dim=1)
        enhanced = self.feature_refine(combined)
        # print(f"Final enhanced features: {enhanced.shape}")
        
        # Return to original format [B, T, C, H, W]
        output = enhanced.permute(0, 2, 1, 3, 4)
        return output.contiguous()

# Reuse your existing EnhancedTemporalSmoothingModule but modify it for binary segmentation
# In models/temporal_components.py

class EnhancedTemporalSmoothingModule(nn.Module):
    """
    Applies temporal smoothing while maintaining consistent segmentation.
    Modified to work with binary segmentation (1 channel) instead of instance segmentation.
    """
    def __init__(
        self,
        channels: int = 1,  # Now default to 1 for binary segmentation
        temporal_kernel: int = 3
    ):
        super().__init__()
        
        # Calculate appropriate number of groups
        num_groups = 1  # For binary segmentation, just use 1 group
        
        # Temporal smoothing
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel//2, 0, 0),
            groups=num_groups
        )
        
        # Feature refinement
        self.segment_refine = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing to segmentation features.
        
        Args:
            features: Segmentation features [B, T, C, H, W] where C is now 1
            
        Returns:
            Temporally smoothed features with same shape
        """
        # Reshape for temporal processing
        B, T, C, H, W = features.shape
        features = features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Apply smoothing
        identity = features
        smoothed = self.temporal_conv(features)
        enhanced = self.segment_refine(smoothed + identity)
        
        # Return to original shape
        return enhanced.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
```

# models\video_instance_decoder.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

from .position_encoding import LeveledPositionEmbeddingSine
from .temporal_components import InstanceMotionModule,InstanceTemporalAttention

class InstanceMemory(nn.Module):
    """
    Memory module that tracks instance features across frames.
    Provides temporal consistency for instance identity.
    """
    def __init__(self, feature_dim: int, num_instances: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_instances = num_instances
        
        # GRU cell for updating instance memory
        self.memory_update = nn.GRUCell(
            input_size=feature_dim,
            hidden_size=feature_dim
        )
        
        # Instance association score
        self.association_score = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, instance_features: torch.Tensor, time_idx: int) -> torch.Tensor:
        """
        Update instance memory with new features.
        
        Args:
            instance_features: Features of current instances [B, N, C]
            time_idx: Current time index
            
        Returns:
            Updated instance memory [B, N, C]
        """
        B, N, C = instance_features.shape
        
        # First frame - initialize memory
        if time_idx == 0:
            return instance_features
        
        # Flatten batch and instance dimensions
        flat_features = instance_features.view(B * N, C)
        
        # Update memory
        updated_memory = self.memory_update(
            flat_features,
            flat_features  # Use current features as hidden state for simplicity
        )
        
        # Reshape back to [B, N, C]
        updated_memory = updated_memory.view(B, N, C)
        
        return updated_memory


class VideoInstanceDecoder(nn.Module):
    """
    Video instance decoder that maintains instance identity across frames.
    Uses query-based attention with explicit temporal handling.
    """
    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 256,
        num_instances: int = 16,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Feature projections for each level
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, kernel_size=1)
            for dim in in_channels
        ])
        
        # Position encodings that match the INPUT feature dimensions
        self.position_encodings = nn.ModuleList([
            LeveledPositionEmbeddingSine(dim)  # Use original input channel dimensions
            for dim in in_channels
        ])
        
        # Instance query embeddings (learnable)
        self.instance_queries = nn.Parameter(
            torch.randn(num_instances, hidden_dim)
        )
        
        # Add InstanceMotionModule for motion modeling
        self.instance_motion = InstanceMotionModule(
            feature_dim=hidden_dim,
            num_instances=num_instances
        )
        
        # Add InstanceTemporalAttention for temporal feature processing
        self.temporal_attention = InstanceTemporalAttention(
            feature_dim=hidden_dim,
            num_instances=num_instances,
            num_heads=num_heads
        )
        
        # Temporal instance memory to track instances across frames
        self.instance_memory = InstanceMemory(
            feature_dim=hidden_dim,
            num_instances=num_instances
        )
        
        # Cross-frame attention to link instances between frames
        self.query_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Mask prediction head
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
    
    def forward(
        self, 
        features: List[List[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of frames with instance tracking.
        
        Args:
            features: List of feature lists for each frame
                     [frames][levels][B, C, H, W]
        
        Returns:
            Dictionary with pred_masks of shape [B, T, N, H, W]
        """
        T = len(features)  # Number of frames
        B = features[0][0].shape[0]  # Batch size
        
        # 1. Apply position encoding and project features for each frame
        projected_features = []
        
        for t in range(T):
            frame_features = features[t]
            # First apply position encoding to input features
            positioned_features = []
            for i, (feat, pos_enc) in enumerate(zip(frame_features, self.position_encodings)):
                pos = pos_enc(feat)  # Generate position encoding
                feat_with_pos = feat + pos  # Add position encoding to features
                positioned_features.append(feat_with_pos)
            
            # Then project to hidden dimension
            projected = [
                proj(feat) for feat, proj in zip(positioned_features, self.feature_projections)
            ]
            projected_features.append(projected)
        
        # 2. Process through motion module and temporal attention if T > 1
        if T > 1:
            # Extract highest resolution features for motion processing
            motion_input = torch.stack([
                projected_features[t][0] for t in range(T)
            ], dim=1)  # [B, T, C, H, W]
            
            # Apply motion module
            motion_features, motion_field = self.instance_motion(motion_input)
            
            # Apply temporal attention
            attended_features = self.temporal_attention(motion_features, motion_field)
            
            # Update the projected features with motion-aware information
            for t in range(T):
                projected_features[t][0] = attended_features[:, t]
        else:
            motion_field = None
            
        # 3. Process frames sequentially to maintain instance tracking
        all_masks = []
        prev_instances = None
        
        for t in range(T):
            # Use highest resolution features
            feat = projected_features[t][0]  # [B, C, H, W]
            C, H, W = feat.shape[1:]
            
            # Initialize instance queries
            if prev_instances is None:
                # First frame - use learnable queries
                instance_queries = self.instance_queries.unsqueeze(0).expand(B, -1, -1)
            else:
                # Use previous frame instances with attention
                instance_queries, _ = self.query_attention(
                    self.instance_queries.unsqueeze(0).expand(B, -1, -1),
                    prev_instances,
                    prev_instances
                )
            
            # Generate instance features through attention
            flat_features = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            
            # Each instance generates its mask
            instance_masks = []
            instance_features = []
            
            for i in range(instance_queries.size(1)):
                query = instance_queries[:, i:i+1, :]  # [B, 1, C]
                
                # Compute attention between query and features
                attn = torch.bmm(query, flat_features.transpose(1, 2))  # [B, 1, H*W]
                attn = F.softmax(attn / math.sqrt(C), dim=2)
                
                # Generate instance-specific features
                inst_feat = torch.bmm(attn, flat_features)  # [B, 1, C]
                instance_features.append(inst_feat)
                
                # Generate mask for this instance
                mask_feat = inst_feat.view(B, C, 1, 1).expand(-1, -1, H, W)
                mask = self.mask_predictor(mask_feat)
                instance_masks.append(mask)
            
            # Stack all instance masks
            frame_masks = torch.cat(instance_masks, dim=1)  # [B, N, H, W]
            all_masks.append(frame_masks)
            
            # Stack instance features
            frame_features = torch.cat(instance_features, dim=1)  # [B, N, C]
            
            # Update instance memory for next frame
            prev_instances = self.instance_memory(frame_features, t)
        
        # 4. Stack masks from all frames [B, T, N, H, W]
        pred_masks = torch.stack(all_masks, dim=1)
        
        # 5. Return results with motion field if available
        if motion_field is not None:
            return {
                "pred_masks": pred_masks,
                "motion_field": motion_field
            }
        else:
            return {"pred_masks": pred_masks}
```

# models\video_model.py

```py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F

from .backbone import BackboneEncoder
from .video_instance_decoder import VideoInstanceDecoder
from .temporal_components import EnhancedTemporalSmoothingModule

# In models/video_model.py, modify the VideoMambaSegmentation class

class VideoMambaSegmentation(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.feature_dim = config['hidden_dims'][-1]
        # Remove num_instances parameter since we're doing binary segmentation now
        
        # Initialize backbone (keep this as is)
        self.backbone = BackboneEncoder(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            d_state=config['d_state'],
            temporal_window=config['temporal_window'],
            dropout=config.get('dropout', 0.1),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2)
        )
        
        # Replace instance decoder with binary segmentation decoder
        self.seg_head = BinarySegmentationHead(
            in_channels=config['hidden_dims'],
            hidden_dim=256
        )
        
        # Temporal smoothing can be reused but simplified
        self.temporal_smooth = EnhancedTemporalSmoothingModule(
            channels=1,  # Now only one channel for binary segmentation
            temporal_kernel=3
        )
    
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with proper dimension handling for video data.
        
        Args:
            x: Input tensor [B, T, C, H, W] representing a video sequence
                
        Returns:
            Dictionary containing model outputs
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Process through binary segmentation head
        logits = self.seg_head(backbone_features)
        
        # Reshape to add temporal dimension [B, T, 1, H, W]
        logits = logits.view(B, T, 1, logits.shape[2], logits.shape[3])
        
        # Apply temporal smoothing
        smoothed_logits = self.temporal_smooth(logits)
        
        # Return dictionary with both logits for loss computation and probabilities for visualization
        return {
            'pred_masks': torch.sigmoid(smoothed_logits),  # For visualization and evaluation
            'logits': smoothed_logits  # For loss computation
        }

# Add this to models/video_model.py or create a new file

class BinarySegmentationHead(nn.Module):
    def __init__(self, in_channels: List[int], hidden_dim: int = 256):
        super().__init__()
        
        # Feature projections from each backbone level
        self.projections = nn.ModuleList([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
            for in_dim in in_channels
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(in_channels), hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction layer (just 1 channel for binary segmentation)
        self.predictor = nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Project and resize all feature levels to the highest resolution
        proj_features = []
        
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            projected = proj(feat)
            
            # Resize to match the size of the highest resolution feature
            if i > 0:  # Skip the first (highest resolution) feature
                target_size = features[0].shape[-2:]
                projected = F.interpolate(projected, size=target_size, mode='bilinear', align_corners=False)
            
            proj_features.append(projected)
        
        # Concatenate all feature levels
        fused_features = torch.cat(proj_features, dim=1)
        
        # Apply feature fusion
        fused = self.fusion(fused_features)
        
        # Final prediction
        logits = self.predictor(fused)
        
        return logits

def build_model(config: Dict) -> VideoMambaSegmentation:
    """
    Builds and initializes the model from config.
    Handles both nested and flat configuration formats.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
        
    # Handle nested configuration
    model_config = config.get('model', config)
    
    # Verify required parameters
    required_params = ['input_dim', 'hidden_dims', 'd_state']
    missing_params = [param for param in required_params if param not in model_config]
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Create model
    return VideoMambaSegmentation(model_config)
```

# prediction_visualization.png

This is a binary file of the type: Image

# requirements.txt

```txt
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
_libgcc_mutex=0.1=main
_openmp_mutex=5.1=1_gnu
absl-py=2.1.0=pypi_0
antlr4-python3-runtime=4.9.3=pypi_0
black=25.1.0=pypi_0
bzip2=1.0.8=h5eee18b_6
ca-certificates=2024.12.31=h06a4308_0
certifi=2024.12.14=pypi_0
charset-normalizer=3.4.1=pypi_0
click=8.1.8=pypi_0
cloudpickle=3.1.1=pypi_0
contourpy=1.3.1=pypi_0
cycler=0.12.1=pypi_0
detectron2=0.6=pypi_0
einops=0.8.0=pypi_0
exceptiongroup=1.2.2=pypi_0
filelock=3.13.1=pypi_0
fonttools=4.55.8=pypi_0
fsspec=2024.2.0=pypi_0
fvcore=0.1.5.post20221221=pypi_0
grpcio=1.70.0=pypi_0
huggingface-hub=0.27.1=pypi_0
hydra-core=1.3.2=pypi_0
idna=3.10=pypi_0
iniconfig=2.0.0=pypi_0
iopath=0.1.9=pypi_0
jinja2=3.1.3=pypi_0
kiwisolver=1.4.8=pypi_0
ld_impl_linux-64=2.40=h12ee557_0
libffi=3.4.4=h6a678d5_1
libgcc-ng=11.2.0=h1234567_1
libgomp=11.2.0=h1234567_1
libstdcxx-ng=11.2.0=h1234567_1
libuuid=1.41.5=h5eee18b_0
mamba-ssm=2.2.4=pypi_0
markdown=3.7=pypi_0
markupsafe=2.1.5=pypi_0
matplotlib=3.10.0=pypi_0
mpmath=1.3.0=pypi_0
multiscaledeformableattention=1.0=pypi_0
mypy-extensions=1.0.0=pypi_0
ncurses=6.4=h6a678d5_0
networkx=3.2.1=pypi_0
ninja=1.11.1.3=pypi_0
numpy=1.26.3=pypi_0
nvidia-cublas-cu12=12.4.5.8=pypi_0
nvidia-cuda-cupti-cu12=12.4.127=pypi_0
nvidia-cuda-nvrtc-cu12=12.4.127=pypi_0
nvidia-cuda-runtime-cu12=12.4.127=pypi_0
nvidia-cudnn-cu12=9.1.0.70=pypi_0
nvidia-cufft-cu12=11.2.1.3=pypi_0
nvidia-curand-cu12=10.3.5.147=pypi_0
nvidia-cusolver-cu12=11.6.1.9=pypi_0
nvidia-cusparse-cu12=12.3.1.170=pypi_0
nvidia-nccl-cu12=2.21.5=pypi_0
nvidia-nvjitlink-cu12=12.4.127=pypi_0
nvidia-nvtx-cu12=12.4.127=pypi_0
omegaconf=2.3.0=pypi_0
opencv-python=4.11.0.86=pypi_0
openssl=3.0.15=h5eee18b_0
packaging=24.2=pypi_0
pathspec=0.12.1=pypi_0
pillow=10.2.0=pypi_0
pip=24.2=pypi_0
platformdirs=4.3.6=pypi_0
pluggy=1.5.0=pypi_0
portalocker=3.1.1=pypi_0
protobuf=5.29.3=pypi_0
pycocotools=2.0.8=pypi_0
pyparsing=3.2.1=pypi_0
pytest=8.3.4=pypi_0
python=3.10.16=he870216_1
python-dateutil=2.9.0.post0=pypi_0
pyyaml=6.0.2=pypi_0
readline=8.2=h5eee18b_0
regex=2024.11.6=pypi_0
requests=2.32.3=pypi_0
safetensors=0.5.2=pypi_0
scipy=1.15.1=pypi_0
setuptools=75.1.0=pypi_0
six=1.17.0=pypi_0
sqlite=3.45.3=h5eee18b_0
sympy=1.13.1=pypi_0
tabulate=0.9.0=pypi_0
tensorboard=2.18.0=pypi_0
tensorboard-data-server=0.7.2=pypi_0
termcolor=2.5.0=pypi_0
timm=1.0.14=pypi_0
tk=8.6.14=h39e8969_0
tokenizers=0.21.0=pypi_0
tomli=2.2.1=pypi_0
torch=2.5.1+cu124=pypi_0
torchvision=0.20.1+cu124=pypi_0
tqdm=4.67.1=pypi_0
transformers=4.48.1=pypi_0
triton=3.1.0=pypi_0
typing-extensions=4.9.0=pypi_0
tzdata=2025a=h04d1e81_0
urllib3=2.3.0=pypi_0
werkzeug=3.1.3=pypi_0
wheel=0.44.0=pypi_0
xz=5.4.6=h5eee18b_1
yacs=0.1.8=pypi_0
zlib=1.2.13=h5eee18b_1

```

# tests\davis_test_results\breakdance_batch_0.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_1.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_2.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_3.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_4.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_5.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_6.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_7.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_8.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_9.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_10.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_11.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_12.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_13.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_14.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_15.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_16.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_17.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_18.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_19.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_20.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_21.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_22.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_23.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_24.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_25.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_26.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_27.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_28.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_29.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_30.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_31.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_32.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_33.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_34.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_35.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_36.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_37.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_38.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_39.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_40.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_41.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_42.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_43.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_44.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_45.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_46.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_47.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_48.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_49.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_50.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_51.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_52.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_53.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_54.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_55.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_56.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_57.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_58.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_59.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_60.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_61.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_62.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_63.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_64.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_65.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_66.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_67.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_68.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_69.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_70.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_71.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_72.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_73.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_74.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_75.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_76.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_77.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_78.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_79.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_80.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_81.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_82.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_83.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_84.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_85.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_86.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_87.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_88.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_89.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_90.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_91.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_92.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_93.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_94.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_95.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_96.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_97.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_98.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_99.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_100.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_101.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_102.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_103.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_104.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_105.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_106.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_107.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_108.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_109.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_110.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_111.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_112.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_113.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_114.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_115.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_116.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_117.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_118.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_119.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_120.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_121.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_122.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_123.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_124.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_125.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_126.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_127.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_128.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_129.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_130.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_131.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_132.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_133.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_134.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_135.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_136.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_137.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_138.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_139.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_140.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_141.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_142.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_143.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_144.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_145.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_146.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_147.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_148.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_149.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_150.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_151.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_152.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_153.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_154.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_155.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_156.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_157.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_158.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_159.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_160.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_161.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_162.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_163.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_164.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_165.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_166.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_167.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_168.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_169.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_170.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_171.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_172.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_173.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_174.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_175.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_176.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_177.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_178.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_179.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_180.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_181.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_182.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_183.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_184.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_185.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_186.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_187.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_188.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_189.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_190.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_191.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_192.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_193.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_194.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_195.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_196.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_197.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_198.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_199.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_200.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_201.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_202.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_203.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_204.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_205.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_206.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_207.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_208.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_209.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_210.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_211.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_212.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_213.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_214.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_215.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_216.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_217.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_218.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_219.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_220.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_221.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_222.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_223.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_224.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_225.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_226.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_227.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_228.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_229.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_230.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_231.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_232.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_233.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_234.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_235.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_236.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_237.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_238.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_239.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_240.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_241.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_242.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_243.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_244.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_245.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_246.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_247.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_248.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_249.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_250.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_251.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_252.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_253.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_254.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_255.png

This is a binary file of the type: Image

# tests\davis_test_results\breakdance_batch_256.png

This is a binary file of the type: Image

# tests\davis_visualization\disc-jockey_no_aug.png

This is a binary file of the type: Image

# tests\davis_visualization\disc-jockey.png

This is a binary file of the type: Image

# tests\davis_visualization\dog-gooses_no_aug.png

This is a binary file of the type: Image

# tests\davis_visualization\dog-gooses.png

This is a binary file of the type: Image

# tests\davis_visualization\drone_no_aug.png

This is a binary file of the type: Image

# tests\davis_visualization\drone.png

This is a binary file of the type: Image

# tests\davis_visualization\flamingo_no_aug.png

This is a binary file of the type: Image

# tests\davis_visualization\flamingo.png

This is a binary file of the type: Image

# tests\davis_visualization\stunt.png

This is a binary file of the type: Image

# tests\davis_visualization\upside-down_no_aug.png

This is a binary file of the type: Image

# tests\davis_visualization\upside-down.png

This is a binary file of the type: Image

# tests\motion_field.png

This is a binary file of the type: Image

# tests\requirements.txt

```txt
absl-py==2.1.0
antlr4-python3-runtime==4.9.3
black==25.1.0
certifi==2024.12.14
charset-normalizer==3.4.1
click==8.1.8
cloudpickle==3.1.1
contourpy==1.3.1
cycler==0.12.1
detectron2 @ git+https://github.com/facebookresearch/detectron2.git@9604f5995cc628619f0e4fd913453b4d7d61db3f
einops==0.8.0
exceptiongroup==1.2.2
filelock==3.13.1
fonttools==4.55.8
fsspec==2024.2.0
fvcore==0.1.5.post20221221
grpcio==1.70.0
huggingface-hub==0.27.1
hydra-core==1.3.2
idna==3.10
iniconfig==2.0.0
iopath==0.1.9
Jinja2==3.1.3
kiwisolver==1.4.8
mamba-ssm==2.2.4
Markdown==3.7
MarkupSafe==2.1.5
matplotlib==3.10.0
mpmath==1.3.0
MultiScaleDeformableAttention==1.0
mypy-extensions==1.0.0
networkx==3.2.1
ninja==1.11.1.3
numpy==1.26.3
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
omegaconf==2.3.0
opencv-python==4.11.0.86
packaging==24.2
pathspec==0.12.1
pillow==10.2.0
platformdirs==4.3.6
pluggy==1.5.0
portalocker==3.1.1
protobuf==5.29.3
pycocotools==2.0.8
pyparsing==3.2.1
pytest==8.3.4
python-dateutil==2.9.0.post0
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
safetensors==0.5.2
scipy==1.15.1
six==1.17.0
sympy==1.13.1
tabulate==0.9.0
tensorboard==2.18.0
tensorboard-data-server==0.7.2
termcolor==2.5.0
timm==1.0.14
tokenizers==0.21.0
tomli==2.2.1
torch==2.5.1+cu124
torchvision==0.20.1+cu124
tqdm==4.67.1
transformers==4.48.1
triton==3.1.0
typing_extensions==4.9.0
urllib3==2.3.0
Werkzeug==3.1.3
yacs==0.1.8

```

# tests\test_backbone.py

```py
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.backbone import BackboneEncoder

def test_backbone():
    print("Testing backbone with dummy data...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 2
    time_steps = 2
    channels = 3
    height = 64
    width = 64
    
    # Move data to correct device
    video_frames = torch.randn(batch_size, time_steps, channels, height, width).to(device)
    flows = torch.randn(batch_size, time_steps-1, 2, height, width).to(device)
    
    print(f"Input shape: {video_frames.shape}")
    
    # Initialize backbone on GPU
    backbone = BackboneEncoder(
        input_dim=channels,
        hidden_dims=[32, 64, 128],
        d_state=16,
        temporal_window=2,
        dropout=0.1
    ).to(device)
    
    try:
        b, t, c, h, w = video_frames.shape
        reshaped_frames = video_frames.reshape(-1, c, h, w)
        
        features = backbone(reshaped_frames, motion_info=flows)
        
        print("\nOutput feature shapes at each scale:")
        for i, feat in enumerate(features):
            feat = feat.view(b, t, *feat.shape[1:])
            print(f"Scale {i + 1}: {feat.shape}")
        
        print("\nBackbone test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during backbone test: {str(e)}")
        return False

def test_components():
    print("\nTesting individual components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    
    from models.backbone import CNNBackbone
    print("\nTesting CNN Backbone...")
    try:
        x = torch.randn(batch_size, channels, height, width).to(device)
        cnn = CNNBackbone(input_dim=channels, hidden_dims=[32, 64, 128]).to(device)
        cnn_features = cnn(x)
        print("CNN Backbone output shapes:")
        for i, feat in enumerate(cnn_features):
            print(f"Layer {i + 1}: {feat.shape}")
    except Exception as e:
        print(f"CNN Backbone error: {str(e)}")
    
    from models.backbone import VideoMambaBlock
    print("\nTesting Mamba Block...")
    try:
        mamba = VideoMambaBlock(d_model=32, d_state=16).to(device)
        mamba_in = torch.randn(batch_size, 32, height, width).to(device)
        mamba_out, state = mamba(mamba_in)
        print(f"Mamba Block output shape: {mamba_out.shape}")
    except Exception as e:
        print(f"Mamba Block error: {str(e)}")
    
    from models.backbone import TemporalFeatureBank
    print("\nTesting Temporal Feature Bank...")
    try:
        tfb = TemporalFeatureBank(feature_dim=32, window_size=2).to(device)
        tfb_in = torch.randn(batch_size, 32, height, width).to(device)
        confidence = torch.ones(batch_size, 1, height, width).to(device)
        tfb.update(tfb_in, confidence)
        tfb_out = tfb.get_temporal_context(tfb_in)
        print(f"Temporal Feature Bank output shape: {tfb_out.shape}")
    except Exception as e:
        print(f"Temporal Feature Bank error: {str(e)}")

if __name__ == "__main__":
    test_backbone()
    test_components()
```

# tests\test_davis_integration.py

```py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from losses.video_instance_loss import VideoInstanceSegmentationLoss

def test_davis_integration():
    """Test model with actual DAVIS data."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    model = build_model(config).to(device)
    model.eval()  # Set to evaluation mode
    
    # Create dataloader with a single specific sequence
    transform = VideoSequenceAugmentation(
        img_size=(240, 320),
        normalize=True,
        train=False
    )
    
    try:
        dataloader = build_davis_dataloader(
            root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
            split='val',
            batch_size=1,
            img_size=(240, 320),
            sequence_length=4,
            specific_sequence="breakdance",  # Test with a specific sequence
            transform=transform
        )
        
        # Create loss function
        criterion = VideoInstanceSegmentationLoss()
        
        # Process one batch
        for batch in dataloader:
            # Move data to device
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            masks = batch.get('masks')
            if masks is not None:
                masks = masks.to(device)  # [B, T, H, W]
            
            print(f"Input frames shape: {frames.shape}")
            if masks is not None:
                print(f"Ground truth masks shape: {masks.shape}")
            
            # Forward pass with gradient tracking disabled
            with torch.no_grad():
                outputs = model(frames)
            
            print(f"Output pred_masks shape: {outputs['pred_masks'].shape}")
            
            # Compute loss if ground truth available
            if masks is not None:
                loss_dict = criterion(outputs, {'masks': masks})
                print(f"Loss values: {loss_dict}")
            
            # Only process one batch
            break
        
        print("DAVIS integration test passed!")
        return True
        
    except Exception as e:
        print(f"Error during DAVIS integration test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_davis_integration()
```

# tests\test_davis_realistic.py

```py
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
```

# tests\test_davis.py

```py


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
```

# tests\test_decoder.py

```py
import torch
import sys
from pathlib import Path

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Update imports to match actual implementation
from models.decoder import (
    SegmentationHead,
    MambaMask2FormerDecoder,
    EnhancedTemporalSmoothingModule
)

def test_decoder():
    """
    Tests the complete decoder pipeline including Mask2Former integration,
    temporal smoothing, and final mask generation.
    """
    print("Testing decoder with dummy data...")
    
    # Set up device for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data with realistic dimensions
    batch_size = 2
    time_steps = 2
    channels = [32, 64, 128]
    height = 64
    width = 64
    num_classes = 21
    
    # Create multi-scale features
    features = [
        torch.randn(batch_size * time_steps, c, height, width).to(device)
        for c in channels
    ]
    
    # Create optical flow between consecutive frames
    flows = torch.randn(batch_size, time_steps-1, 2, height, width).to(device)
    
    # Initialize decoder
    mask2former_config = {
        'hidden_dim': 256,
        'num_queries': 100,
        'nheads': 8,
        'dim_feedforward': 1024,
        'dec_layers': 6,
        'mask_dim': 256,
        'enforce_input_project': False
    }
    
    try:
        decoder = SegmentationHead(
            in_channels=channels,
            mask2former_config=mask2former_config,
            num_classes=num_classes
        ).to(device)
        
        # Forward pass
        outputs = decoder(features, flows)
        
        # Verify output is dictionary during inference
        assert isinstance(outputs, dict), "Output should be a dictionary"
        assert 'pred_masks' in outputs, "Output should contain 'pred_masks'"
        
        masks = outputs['pred_masks']
        print(f"\nOutput mask shape: {masks.shape}")
        
        # Verify output dimensions
        expected_shape = (batch_size * time_steps, num_classes, height, width)
        assert masks.shape == expected_shape, f"Expected shape {expected_shape}, got {masks.shape}"
        
        print("\nDecoder test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during decoder test: {str(e)}")
        raise e  # Re-raise to see full traceback

if __name__ == "__main__":
    test_decoder()
```

# tests\test_evaluation_metrics.py

```py
def test_evaluation_metrics():
    """Test the evaluation metrics."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    model = build_model(config).to(device)
    model.eval()  # Set to evaluation mode
    
    try:
        # Create dataloader
        transform = VideoSequenceAugmentation(
            img_size=(240, 320),
            normalize=True,
            train=False
        )
        
        # Test on multiple sequences
        sequences = ["breakdance", "camel", "car-roundabout"]
        all_predictions = []
        all_ground_truths = []
        sequence_names = []
        
        for sequence in sequences:
            try:
                dataloader = build_davis_dataloader(
                    root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
                    split='val',
                    batch_size=1,
                    img_size=(240, 320),
                    sequence_length=4,
                    specific_sequence=sequence,
                    transform=transform
                )
                
                # Process one batch
                for batch in dataloader:
                    # Skip if no ground truth
                    if 'masks' not in batch:
                        continue
                    
                    # Move data to device
                    frames = batch['frames'].to(device)  # [B, T, C, H, W]
                    gt_masks = batch['masks'].to(device)  # [B, T, H, W]
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(frames)
                    
                    pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
                    
                    # Save predictions and ground truth
                    all_predictions.append(pred_masks[0])  # Remove batch dimension
                    all_ground_truths.append(gt_masks[0])  # Remove batch dimension
                    sequence_names.append(sequence)
                    
                    # Only process one batch per sequence
                    break
            except Exception as e:
                print(f"Error processing sequence {sequence}: {str(e)}")
                continue
        
        # Skip evaluation if no sequences were processed
        if not all_predictions:
            print("No sequences processed, skipping evaluation")
            return False
        
        # Create evaluator
        evaluator = DAVISEvaluator()
        
        # Evaluate all sequences
        results = evaluator.evaluate(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            sequence_names=sequence_names
        )
        
        # Print results
        evaluator.print_results(results)
        
        print("Evaluation metrics test completed!")
        return True
        
    except Exception as e:
        print(f"Error during evaluation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

```

# tests\test_instance_segmentation.py

```py
import torch
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import build_model

def create_test_config(small_test=False):
    """
    Creates a configuration for testing. Can create either a full-size or small test configuration.
    
    Args:
        small_test: If True, creates a smaller configuration for dimension testing
    """
    if small_test:
        return {
            'model': {
                'input_dim': 3,
                'hidden_dims': [16, 32, 64],  # Smaller for testing
                'd_state': 16,
                'temporal_window': 4,
                'dropout': 0.1,
                'd_conv': 4,
                'expand': 2,
                'mask2former': {
                    'hidden_dim': 128,
                    'num_queries': 16,
                    'nheads': 4,
                    'dim_feedforward': 256,
                    'dec_layers': 3,
                    'mask_dim': 128,
                    'enforce_input_project': False
                }
            }
        }
    else:
        return {
            'model': {
                'input_dim': 3,
                'hidden_dims': [32, 64, 128],
                'd_state': 16,
                'temporal_window': 4,
                'dropout': 0.1,
                'd_conv': 4,
                'expand': 2,
                'mask2former': {
                    'hidden_dim': 256,
                    'num_queries': 16,
                    'nheads': 4,
                    'dim_feedforward': 512,
                    'dec_layers': 6,
                    'mask_dim': 256,
                    'enforce_input_project': False
                }
            }
        }

def test_memory_usage(device):
    """Print current GPU memory usage if using CUDA."""
    if device.type == 'cuda':
        current_mem = torch.cuda.memory_allocated() / 1e9
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Current GPU memory: {current_mem:.2f} GB")
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

def test_dimensions():
    """
    Test if dimensions are correct throughout the pipeline.
    This function uses a smaller model and input size to verify dimension handling.
    """
    print("\nStarting dimension testing...")
    
    # Create small test input
    B, T, C = 1, 4, 3
    H, W = 128, 128
    x = torch.randn(B, T, C, H, W)
    
    # Use smaller configuration for dimension testing
    config = create_test_config(small_test=True)
    
    try:
        # Create and move model to available device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(config).to(device)
        model.eval()
        x = x.to(device)
        
        print(f"\nTesting with configuration:")
        print(f"- Input shape: {x.shape}")
        print(f"- Hidden dimensions: {config['model']['hidden_dims']}")
        print(f"- Number of queries: {config['model']['mask2former']['num_queries']}")
        
        with torch.no_grad():
            # Process through model
            outputs = model(x)
            pred_masks = outputs['pred_masks']
            
            # Verify output dimensions
            expected_instances = config['model']['mask2former']['num_queries']
            B_out, N_out, H_out, W_out = pred_masks.shape
            
            print("\nOutput dimensions:")
            print(f"- Batch size (B*T): {B_out} (expected {B*T})")
            print(f"- Number of instances: {N_out} (expected {expected_instances})")
            print(f"- Height: {H_out} (expected {H})")
            print(f"- Width: {W_out} (expected {W})")
            
            # Verify dimension relationships
            assert B_out == B * T, "Batch dimension mismatch"
            assert N_out == expected_instances, "Instance dimension mismatch"
            assert H_out == H, "Height mismatch"
            assert W_out == W, "Width mismatch"
            
            print("\nDimension test passed successfully!")
            return True
            
    except Exception as e:
        print(f"\nDimension test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_instance_segmentation():
    """
    Tests the complete instance segmentation pipeline with full-size configuration.
    """
    print("\nTesting instance segmentation with temporal components...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create model with full-size config
        config = create_test_config(small_test=False)
        print("\nCreating model...")
        model = build_model(config).to(device)
        model.eval()
        test_memory_usage(device)
        
        # Create test dimensions
        batch_size = 1
        sequence_length = 4
        height = 240
        width = 320
        
        print(f"\nTest dimensions:")
        print(f"- Batch size: {batch_size}")
        print(f"- Sequence length: {sequence_length}")
        print(f"- Resolution: {height}x{width}")
        print(f"- Number of queries: {config['model']['mask2former']['num_queries']}")
        
        # Create input tensor
        print("\nCreating input tensor...")
        video_input = torch.randn(
            batch_size, sequence_length, 3, height, width
        ).to(device)
        test_memory_usage(device)
        
        # Run forward pass
        print("\nRunning forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(video_input)
        
        inference_time = time.time() - start_time
        fps = sequence_length / inference_time
        
        print(f"\nPerformance metrics:")
        print(f"- Inference time: {inference_time:.3f} seconds")
        print(f"- Frames per second: {fps:.2f}")
        test_memory_usage(device)
        
        # Verify outputs
        print("\nVerifying outputs...")
        pred_masks = outputs['pred_masks']
        print(f"- Prediction shape: {pred_masks.shape}")
        print(f"- Value range: [{pred_masks.min():.3f}, {pred_masks.max():.3f}]")
        print(f"- Memory after forward pass:")
        test_memory_usage(device)
        
        print("\nInstance segmentation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First run dimension test with smaller configuration
    if not test_dimensions():
        print("\nDimension test failed - skipping full model test")
        sys.exit(1)
        
    # If dimensions are correct, run full model test
    print("\nDimension test passed - proceeding with full model test")
    test_instance_segmentation()
```

# tests\test_model.py

```py
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import VideoMambaSegmentation, build_model

def create_sample_config():
    """Creates a sample configuration for testing the model.
    
    The configuration includes all necessary parameters for both the backbone
    and decoder components, ensuring proper initialization of the full pipeline.
    """
    return {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'dropout': 0.1,
        'd_conv': 4,
        'expand': 2,
        'num_classes': 21,
        'mask2former': {
            'hidden_dim': 256,
            'num_queries': 100,
            'nheads': 8,
            'dim_feedforward': 1024,
            'dec_layers': 6,
            'mask_dim': 256,
            'enforce_input_project': False
        }
    }

def test_model_initialization():
    """Tests if the model initializes correctly with the sample configuration."""
    print("\nTesting model initialization...")
    
    config = create_sample_config()
    model = build_model(config)
    
    assert hasattr(model, 'backbone'), "Backbone is missing"
    assert hasattr(model, 'seg_head'), "Segmentation head is missing"
    
    print("Model initialization test passed!")


def test_model_forward():
    """Tests the model's forward pass with proper tensor shape handling."""
    print("\nTesting model forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = create_sample_config()
    model = build_model(config).to(device)
    model.eval()
    
    # Create test data
    batch_size = 2
    time_steps = 2
    channels = 3
    height = 64
    width = 64
    
    # Create input tensors with proper shapes
    dummy_input = torch.randn(batch_size, time_steps, channels, height, width).to(device)
    mask_features = torch.randn(batch_size, channels, time_steps, height, width).to(device)
    
    try:
        print("Testing inference mode...")
        with torch.no_grad():
            # Forward pass with both inputs
            outputs = model(dummy_input, mask_features)  # No keyword argument
            
            # Verify outputs
            assert 'pred_masks' in outputs, "pred_masks missing from outputs"
            pred_masks = outputs['pred_masks']
            
            # Expected shape after temporal flattening
            expected_shape = (batch_size * time_steps, config['num_classes'], height, width)
            print(f"Expected shape: {expected_shape}")
            print(f"Actual shape: {pred_masks.shape}")
            
            assert pred_masks.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {pred_masks.shape}"
            )
            
            # Verify output values
            assert torch.isfinite(pred_masks).all(), "Output contains inf or nan values"
            
            print("Forward pass test completed successfully!")
            return True
            
    except Exception as e:
        print(f"\nError during forward pass test: {str(e)}")
        return False

def test_model_output_values():
    """Tests if the model's outputs have reasonable values within expected ranges."""
    print("\nTesting model output values...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_sample_config()
    model = build_model(config).to(device)
    model.eval()
    
    batch_size = 2
    time_steps = 2
    dummy_input = torch.randn(batch_size, time_steps, 3, 64, 64).to(device)
    mask_features = torch.randn(batch_size, 3, time_steps, 64, 64).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input, mask_features=mask_features)
        masks = outputs['pred_masks']
        
        # Verify values are within reasonable range (e.g., post-sigmoid would be 0-1)
        assert torch.all((masks >= 0) & (masks <= 1)), "Mask values outside expected range"
        print("Output values test passed!")
        return True

def run_all_tests():
    """Runs all model tests in sequence with proper error handling."""
    try:
        test_model_initialization()
        test_model_forward()
        test_model_output_values()
        print("\nAll tests passed successfully!")
        return True
    except Exception as e:
        print(f"\nTest suite failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    run_all_tests()
```

# tests\test_realistic.py

```py
import torch
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import build_model

def create_realistic_config():
    """Creates configuration with memory-efficient parameters."""
    return {
        'model': {  # Nest model parameters under 'model' key
            'input_dim': 3,
            'hidden_dims': [32, 64, 128],  # Reduced channel dimensions
            'd_state': 16,
            'temporal_window': 4,          # Reduced temporal window
            'dropout': 0.1,
            'd_conv': 4,
            'expand': 2,
            'num_classes': 21,
            'mask2former': {
                'hidden_dim': 256,
                'num_queries': 100,
                'nheads': 8,
                'dim_feedforward': 1024,    # Reduced feedforward dimension
                'dec_layers': 6,
                'mask_dim': 256,
                'enforce_input_project': False
            }
        },
        # Additional configuration sections
        'training': {
            'epochs': 100,
            'batch_size': 1,
            'mixed_precision': True
        },
        'dataset': {
            'img_size': [240, 320],  # Half resolution for testing
            'sequence_length': 4
        }
    }

def test_realistic_scenario():
    """Tests the model with memory-efficient but still realistic parameters."""
    print("\nTesting memory-efficient video segmentation scenario...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with efficient config
    config = create_realistic_config()
    model = build_model(config).to(device)
    model.eval()
    
    # Memory-efficient dimensions but still realistic ratio
    batch_size = 1          # Process 1 video at a time
    sequence_length = 4     # 4 frames per sequence
    height = config['dataset']['img_size'][0]  # Use height from config
    width = config['dataset']['img_size'][1]   # Use width from config
    channels = config['model']['input_dim']    # Use channels from config
    
    try:
        # Create input tensors
        print("\nPreparing input tensors...")
        print(f"Input dimensions:")
        print(f"- Batch size: {batch_size} (videos)")
        print(f"- Sequence length: {sequence_length} frames")
        print(f"- Resolution: {height}x{width}")
        print(f"- Channels: {channels} (RGB)")
        
        # Simulate video input
        video_input = torch.randn(batch_size, sequence_length, channels, height, width).to(device)
        
        # Simulate mask features
        mask_features = torch.randn(batch_size, channels, sequence_length, height, width).to(device)
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            print(f"\nInitial GPU memory used: {start_mem/1e9:.2f} GB")
        
        # Time the forward pass
        print("\nRunning forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(video_input, mask_features)
        
        # Calculate timing
        inference_time = time.time() - start_time
        
        # Memory statistics
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            mem_diff = peak_mem - start_mem
            print(f"\nMemory Usage:")
            print(f"- Peak memory: {peak_mem/1e9:.2f} GB")
            print(f"- Memory increase: {mem_diff/1e9:.2f} GB")
        
        # Performance metrics
        fps = (batch_size * sequence_length)/inference_time
        print(f"\nPerformance Metrics:")
        print(f"- Total inference time: {inference_time:.3f} seconds")
        print(f"- Frames per second: {fps:.2f}")
        
        # Analyze outputs
        pred_masks = outputs['pred_masks']
        print(f"\nOutput Analysis:")
        print(f"- Prediction shape: {pred_masks.shape}")
        
        print("\nMemory-efficient scenario test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        return False

if __name__ == "__main__":
    test_realistic_scenario()
```

# tests\test_simple_backbone.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLayerNorm(nn.Module):
    """
    Custom LayerNorm that handles temporal data while maintaining dimensional consistency.
    This ensures proper normalization across channels while preserving temporal information.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        # x input shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Rearrange for channel normalization
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        x = x.reshape(-1, C)  # Combine all dimensions except channels
        
        # Apply normalization
        x = self.norm(x)
        
        # Restore original shape
        x = x.view(B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)  # Back to [B, C, T, H, W]
        return x

class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Spatial processing
        self.spatial_proj = nn.Conv2d(d_model, d_model, 1)
        
        # Temporal processing
        self.temporal_proj = nn.Conv3d(
            d_model, d_model,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        
        # Here's the key change: we're using d_model in both state projections
        # because our features are in d_model dimension space
        self.state_proj = nn.Linear(d_state, d_model)
        # This projection now goes from d_model to d_state directly
        self.state_update = nn.Linear(d_model, d_state)
        
    def forward(self, x, state=None):
        batch_size, time_steps, channels, height, width = x.shape
        
        if state is None:
            state = self.init_state(batch_size).to(x.device)
        
        # Project state
        state_features = self.state_proj(state)  # [B, d_model]
        state_features = state_features.view(batch_size, self.d_model, 1, 1, 1)
        state_features = state_features.expand(-1, -1, time_steps, height, width)
        
        # Process temporal dimension
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        temporal_features = self.temporal_proj(x)
        
        # Process each timestep
        output = []
        for t in range(time_steps):
            curr_features = temporal_features[:, :, t]  # [B, C, H, W]
            spatial_features = self.spatial_proj(curr_features)
            combined = spatial_features + state_features[:, :, t]
            output.append(combined)
        
        output = torch.stack(output, dim=1)  # [B, T, C, H, W]
        
        # Here's the critical fix for dimension mismatch:
        features_pooled = output.mean([-2, -1])  # Average over spatial dimensions [B, T, C]
        features_mean = features_pooled.mean(1)   # Average over temporal dimension [B, C]
        new_state = self.state_update(features_mean)  # Project from d_model to d_state [B, d_state]
        
        return output, new_state

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.d_state)
        
class TemporalCNNBackbone(nn.Module):
    """
    CNN backbone that properly handles temporal information and multi-scale features
    while maintaining dimensional consistency throughout the network.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        
        for dim in hidden_dims:
            self.layers.append(nn.Sequential(
                # 3D convolution for spatiotemporal processing
                nn.Conv3d(
                    curr_dim, dim,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1)
                ),
                TemporalLayerNorm(dim),
                nn.ReLU(inplace=True)
            ))
            curr_dim = dim
            
    def forward(self, x):
        """
        Forward pass that maintains temporal dimension throughout processing
        Args:
            x: Input tensor [B, T, C, H, W]
        Returns:
            List of features at different scales, each [B, T, C', H, W]
        """
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        features = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            # Restore temporal dimension order
            features.append(current.permute(0, 2, 1, 3, 4))
            
        return features

def test_simple_backbone():
    print("Starting backbone test...")
    
    # Create test data
    batch_size = 2
    time_steps = 3
    channels = 3
    height = 64
    width = 64
    
    # Create input video
    video = torch.randn(batch_size, time_steps, channels, height, width)
    print(f"\nInput video shape: {video.shape}")
    
    # Initialize models
    hidden_dims = [32, 64, 128]
    temporal_cnn = TemporalCNNBackbone(channels, hidden_dims)
    mamba_blocks = nn.ModuleList([
        SimpleMambaBlock(dim, d_state=16)
        for dim in hidden_dims
    ])
    
    print("\nProcessing video through backbone...")
    
    try:
        # Process through CNN
        cnn_features = temporal_cnn(video)
        
        # Process through Mamba blocks
        final_features = []
        states = [None] * len(mamba_blocks)
        
        for i, (feat, mamba) in enumerate(zip(cnn_features, mamba_blocks)):
            mamba_out, new_state = mamba(feat, states[i])
            states[i] = new_state
            final_features.append(mamba_out)
        
        # Print output shapes
        print("\nOutput feature shapes at each scale:")
        for i, feat in enumerate(final_features):
            print(f"Scale {i + 1}: {feat.shape}")
        
        print("\nTest completed successfully!")
        return final_features
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print(f"Error type: {type(e)}")
        raise e

if __name__ == "__main__":
    features = test_simple_backbone()
```

# tests\test_temporal_components.py

```py
import torch
import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.model import build_model
from models.temporal_components import InstanceTemporalAttention, InstanceMotionModule

def test_instance_motion_module(model, sample_input):
    """
    Tests the instance motion module's ability to track object movement.
    This test verifies that the module can detect and predict motion between frames.
    """
    print("\nTesting Instance Motion Module...")
    B, T, C, H, W = sample_input.shape
    
    # Extract features using backbone
    with torch.no_grad():
        # Reshape input for backbone
        x = sample_input.view(B * T, C, H, W)
        backbone_features = model.backbone(x)
        last_features = backbone_features[-1]
        
        # Process through motion module
        motion_features, motion_field = model.instance_motion(
            last_features.view(B, T, -1, H, W)
        )
        
        # Verify shapes
        print(f"Motion features shape: {motion_features.shape}")
        print(f"Motion field shape: {motion_field.shape}")
        
        # Check motion field properties
        motion_magnitude = torch.norm(motion_field, dim=2)  # [B, T-1, H, W]
        print(f"Average motion magnitude: {motion_magnitude.mean().item():.4f}")
        print(f"Max motion magnitude: {motion_magnitude.max().item():.4f}")
        
        return motion_features, motion_field

def test_temporal_attention(model, sample_input, motion_field=None):
    """
    Tests the temporal attention mechanism's ability to maintain instance consistency.
    This verifies that the attention module properly tracks instances across frames.
    """
    print("\nTesting Temporal Attention...")
    B, T, C, H, W = sample_input.shape
    
    with torch.no_grad():
        # Get backbone features
        x = sample_input.view(B * T, C, H, W)
        backbone_features = model.backbone(x)
        last_features = backbone_features[-1].view(B, T, -1, H, W)
        
        # Apply temporal attention
        attended_features = model.temporal_attention(last_features, motion_field)
        
        # Analyze attention patterns
        print(f"Attended features shape: {attended_features.shape}")
        
        # Check temporal consistency
        feature_diff = torch.norm(
            attended_features[:, 1:] - attended_features[:, :-1],
            dim=2
        ).mean()
        print(f"Temporal consistency score: {feature_diff.item():.4f}")
        
        return attended_features

def test_temporal_smoothing(model, pred_masks):
    """
    Tests the temporal smoothing module's ability to create consistent instance masks.
    This ensures smooth transitions between frames for each instance.
    """
    print("\nTesting Temporal Smoothing...")
    B, T, N, H, W = pred_masks.shape
    
    with torch.no_grad():
        # Apply temporal smoothing
        smoothed_masks = model.temporal_smooth(pred_masks)
        
        # Check smoothing effect
        original_diff = torch.abs(pred_masks[:, 1:] - pred_masks[:, :-1]).mean()
        smoothed_diff = torch.abs(smoothed_masks[:, 1:] - smoothed_masks[:, :-1]).mean()
        
        print(f"Original temporal difference: {original_diff.item():.4f}")
        print(f"Smoothed temporal difference: {smoothed_diff.item():.4f}")
        
        return smoothed_masks

def visualize_motion_field(motion_field, save_path=None):
    """
    Creates a visualization of the motion field to show instance movement.
    This helps us understand how the model tracks object motion.
    """
    B, T, _, H, W = motion_field.shape
    motion_field = motion_field.cpu().numpy()
    
    fig, axes = plt.subplots(B, T-1, figsize=(4*T, 4*B))
    if B == 1:
        axes = axes[None, :]
    
    for b in range(B):
        for t in range(T-1):
            # Create motion field visualization
            U = motion_field[b, t, 0]
            V = motion_field[b, t, 1]
            
            # Subsample for clearer visualization
            step = 8
            Y, X = np.mgrid[0:H:step, 0:W:step]
            U = U[::step, ::step]
            V = V[::step, ::step]
            
            # Plot motion vectors
            axes[b, t].quiver(X, Y, U, V, scale=1, scale_units='xy')
            axes[b, t].set_title(f'Motion t={t}→{t+1}')
            axes[b, t].axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_temporal_components():
    """
    Comprehensive test of all temporal components working together.
    This test verifies the complete temporal processing pipeline.
    """
    print("\nTesting temporal instance segmentation components...")
    
    # Create test configuration with proper nesting structure
    config = {
        'model': {
            'input_dim': 3,
            'hidden_dims': [32, 64, 128],
            'd_state': 16,
            'temporal_window': 4,
            'dropout': 0.1,
            'mask2former': {
                'hidden_dim': 256,
                'num_queries': 16,
                'nheads': 8,
                'dim_feedforward': 1024,
                'dec_layers': 6,
                'mask_dim': 256
            }
        }
    }
    
    # Create sample input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T = 1, 4  # Batch size and sequence length
    H, W = 240, 320  # Spatial dimensions
    x = torch.randn(B, T, 3, H, W).to(device)
    
    # Extract model config from nested structure
    model_config = config['model']
    
    print("\nInitializing model with configuration:")
    print(f"- Input dimension: {model_config['input_dim']}")
    print(f"- Hidden dimensions: {model_config['hidden_dims']}")
    print(f"- Number of instances: {model_config['mask2former']['num_queries']}")
    
    # Build and initialize model
    try:
        model = build_model(model_config).to(device)
        model.eval()
        
        print("\nTesting complete temporal pipeline:")
        
        # 1. Test Instance Motion Module
        print("\nTesting motion estimation...")
        motion_features, motion_field = test_instance_motion_module(model, x)
        
        # Visualize motion field
        visualize_motion_field(motion_field, "motion_field.png")
        print("Motion field visualization saved to motion_field.png")
        
        # 2. Test Temporal Attention
        print("\nTesting temporal attention...")
        attended_features = test_temporal_attention(model, x, motion_field)
        
        # 3. Full forward pass
        print("\nPerforming full forward pass...")
        with torch.no_grad():
            outputs = model(x)
            
            # Get instance masks
            pred_masks = outputs['pred_masks']
            pred_masks = pred_masks.view(B, T, model.num_instances, H, W)
            
            # 4. Test Temporal Smoothing
            print("\nTesting temporal smoothing...")
            smoothed_masks = test_temporal_smoothing(model, pred_masks)
        
        # Report memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"\nMemory Usage:")
            print(f"- Current: {memory_used:.2f} GB")
            print(f"- Peak: {peak_memory:.2f} GB")
        
        # Verify final output properties
        print("\nFinal Output Properties:")
        print(f"- Instance mask shape: {smoothed_masks.shape}")
        print(f"- Value range: [{smoothed_masks.min():.3f}, {smoothed_masks.max():.3f}]")
        
        print("\nTemporal component test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during temporal testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_temporal_components()


```

# tests\test_video_instance_model.py

```py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
from models.video_model import build_model

def test_video_instance_model():
    """Test the video instance segmentation model with random data."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple config
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    
    # Create model and move to device
    model = build_model(config).to(device)
    
    # Create random input data
    batch_size = 1
    sequence_length = 4
    channels = 3
    height = 240
    width = 320
    
    x = torch.randn(batch_size, sequence_length, channels, height, width)
    # Move input to same device as model
    x = x.to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Set model to evaluation mode to avoid batch norm/dropout issues
    model.eval()
    
    # Forward pass with gradient tracking disabled for initial test
    with torch.no_grad():
        try:
            outputs = model(x)
            
            # Check outputs
            assert 'pred_masks' in outputs, "Model should output prediction masks"
            pred_masks = outputs['pred_masks']
            expected_shape = (batch_size, sequence_length, config['num_instances'], height, width)
            print(f"Output pred_masks shape: {pred_masks.shape}")
            assert pred_masks.shape == expected_shape, f"Expected shape {expected_shape}, got {pred_masks.shape}"
            
            print("Basic model test passed!")
        except Exception as e:
            print(f"Error during forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test with gradients if initial test passed
    try:
        # Use a smaller input for gradient test to reduce memory usage
        x_small = torch.randn(1, 2, 3, 120, 160, device=device, requires_grad=True)
        model.train()  # Set to training mode
        outputs = model(x_small)
        pred_masks = outputs['pred_masks']
        loss = pred_masks.mean()  # Simple loss for gradient test
        loss.backward()
        
        print("Gradient test passed!")
        
    except Exception as e:
        print(f"Error during gradient test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_video_instance_model()
```

# tests\test_video_instance.py

```py

```

# tests\test_visualization_and_evaluation.py

```py
# tests/test_visualization_and_evaluation.py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from utils.visualization import VideoSegmentationVisualizer
from utils.evaluation import DAVISEvaluator

def test_with_untrained_model():
    """Test visualization and evaluation with an untrained model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    
    # Create untrained model
    model = build_model(config).to(device)
    model.eval()
    
    # Create dataloader for a specific sequence
    transform = VideoSequenceAugmentation(
        img_size=(240, 320),
        normalize=True,
        train=False
    )
    
    dataloader = build_davis_dataloader(
        root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
        split='val',
        batch_size=1,
        img_size=(240, 320),
        sequence_length=4,
        specific_sequence="breakdance",  # Test with a specific sequence
        transform=transform
    )
    
    # Initialize visualizer and evaluator
    visualizer = VideoSegmentationVisualizer(save_dir="test_visualization")
    evaluator = DAVISEvaluator()
    
    # Process one batch
    for batch in dataloader:
        # Skip batches without ground truth
        if 'masks' not in batch:
            continue
            
        # Get data
        frames = batch['frames'].to(device)
        masks = batch['masks'].to(device)
        sequence_name = batch['sequence'][0]
        
        print(f"Processing sequence: {sequence_name}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(frames)
        
        # Create visualizations
        visualizer.visualize_sequence(
            frames=frames[0].cpu(),
            pred_masks=outputs['pred_masks'][0].cpu(),
            gt_masks=masks[0].cpu(),
            sequence_name=f"{sequence_name}_untrained"
        )
        
        video_path = visualizer.create_video(
            frames=frames[0].cpu(),
            pred_masks=outputs['pred_masks'][0].cpu(),
            gt_masks=masks[0].cpu(),
            sequence_name=f"{sequence_name}_untrained"
        )
        print(f"Created video visualization at {video_path}")
        
        # Calculate metrics
        metrics = evaluator.evaluate(
        predictions=[outputs['pred_masks'][0].cpu()],
        ground_truths=[masks[0].cpu()],
        sequence_names=[sequence_name]
)
        print("\nMetrics with untrained model:")
        print("\nMetrics with untrained model:")
        def print_metrics(metrics_dict, indent=""):
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    print(f"{indent}{key}:")
                    print_metrics(value, indent + "  ")
                elif isinstance(value, (int, float)):
                    print(f"{indent}{key}: {value:.4f}")
                else:
                    print(f"{indent}{key}: {value}")

        print_metrics(metrics)
        
        # Only process one batch
        break
    
    print("Test completed!")
    return True

if __name__ == "__main__":
    test_with_untrained_model()
```

# tests\test_vizualization.py

```py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from losses.video_instance_loss import VideoInstanceSegmentationLoss

# Import our new components
from utils.visualization import VideoSegmentationVisualizer, visualize_instance_tracking
from utils.evaluation import VideoInstanceEvaluator, DAVISEvaluator

def test_visualization_tools():
    """Test the enhanced visualization tools."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    model = build_model(config).to(device)
    model.eval()  # Set to evaluation mode
    
    try:
        # Create dataloader
        transform = VideoSequenceAugmentation(
            img_size=(240, 320),
            normalize=True,
            train=False
        )
        
        dataloader = build_davis_dataloader(
            root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
            split='val',
            batch_size=1,
            img_size=(240, 320),
            sequence_length=4,
            specific_sequence="breakdance",
            transform=transform
        )
        
        # Create visualizer
        visualizer = VideoSegmentationVisualizer(save_dir="visualization_output")
        
        # Process one batch
        for batch in dataloader:
            # Move data to device
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            gt_masks = batch.get('masks')
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)  # [B, T, H, W]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(frames)
            
            pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
            
            # Move tensors to CPU for visualization
            frames = frames.cpu()
            pred_masks = pred_masks.cpu()
            if gt_masks is not None:
                gt_masks = gt_masks.cpu()
            
            # Test single frame visualization
            fig = visualizer.visualize_frame(
                frame=frames[0, 0],
                pred_masks=pred_masks[0, 0],
                gt_mask=gt_masks[0, 0] if gt_masks is not None else None,
                frame_idx=0,
                title="Test Frame Visualization"
            )
            plt.close(fig)
            
            # Test sequence visualization
            sequence_name = batch['sequence'][0]
            _ = visualizer.visualize_sequence(
                frames=frames[0],
                pred_masks=pred_masks[0],
                gt_masks=gt_masks[0] if gt_masks is not None else None,
                sequence_name=sequence_name
            )
            
            # Test video creation
            video_path = visualizer.create_video(
                frames=frames[0],
                pred_masks=pred_masks[0],
                gt_masks=gt_masks[0] if gt_masks is not None else None,
                sequence_name=sequence_name
            )
            print(f"Video saved to: {video_path}")
            
            # Test dashboard
            dashboard = visualizer.create_analysis_dashboard(
                frames=frames[0],
                pred_masks=pred_masks[0],
                gt_masks=gt_masks[0] if gt_masks is not None else None,
                metrics={'J&F': 0.5, 'J_mean': 0.6, 'F_mean': 0.4, 'T_mean': 0.8},
                sequence_name=sequence_name
            )
            plt.close(dashboard)
            
            # Test instance tracking visualization
            track_path = visualize_instance_tracking(
                frames=frames[0],
                pred_masks=pred_masks[0],
                sequence_name=sequence_name,
                save_dir="visualization_output"
            )
            print(f"Instance tracking visualization saved to: {track_path}")
            
            # Only process one batch
            break
            
        print("Visualization tools test completed!")
        return True
        
    except Exception as e:
        print(f"Error during visualization test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


```

# tests\visualize_predictions.py

```py
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation

def visualize_predictions():
    """Visualize model predictions on DAVIS data."""
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_dim': 3,
        'hidden_dims': [32, 64, 128],
        'd_state': 16,
        'temporal_window': 4,
        'num_instances': 16
    }
    model = build_model(config).to(device)
    model.eval()  # Set to evaluation mode
    
    try:
        # Create dataloader
        transform = VideoSequenceAugmentation(
            img_size=(240, 320),
            normalize=True,
            train=False
        )
        
        dataloader = build_davis_dataloader(
            root_path="/mnt/c/Datasets/DAVIS",  # Adjust to your path
            split='val',
            batch_size=1,
            img_size=(240, 320),
            sequence_length=4,
            specific_sequence="breakdance",
            transform=transform
        )
        
        # Process one batch
        for batch in dataloader:
            # Move data to device
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            gt_masks = batch.get('masks')
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)  # [B, T, H, W]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(frames)
            
            pred_masks = outputs['pred_masks']  # [B, T, N, H, W]
            
            # Move tensors to CPU for visualization
            frames = frames.cpu()
            pred_masks = pred_masks.cpu()
            if gt_masks is not None:
                gt_masks = gt_masks.cpu()
            
            # Visualize results
            B, T, N, H, W = pred_masks.shape
            
            # Create visualization figure
            fig, axes = plt.subplots(T, 3, figsize=(15, 5*T))
            
            for t in range(T):
                # Original frame
                frame = frames[0, t].permute(1, 2, 0).numpy()  # [H, W, C]
                # Denormalize if needed
                if frame.max() <= 1.0:
                    frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize
                axes[t, 0].imshow(frame)
                axes[t, 0].set_title(f"Frame {t}")
                
                # Ground truth mask
                if gt_masks is not None:
                    gt = gt_masks[0, t].numpy()
                    axes[t, 1].imshow(gt, cmap='tab20')
                    axes[t, 1].set_title(f"Ground Truth {t}")
                else:
                    axes[t, 1].axis('off')
                
                # Predicted masks
                # Combine instance masks by taking argmax
                pred_combined = torch.zeros((H, W))
                for n in range(N):
                    mask = pred_masks[0, t, n] > 0.5
                    # Assign instance ID (add 1 to avoid conflict with background)
                    pred_combined[mask.squeeze()] = n + 1
                
                axes[t, 2].imshow(pred_combined.numpy(), cmap='tab20')
                axes[t, 2].set_title(f"Prediction {t}")
            
            plt.tight_layout()
            plt.savefig("prediction_visualization.png")
            print(f"Visualization saved to prediction_visualization.png")
            
            # Only process one batch
            break
            
        return True
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    visualize_predictions()
```

# train.py

```py
# train.py

import os
import torch
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from torch.cuda.amp import GradScaler

# Import project components
from utils.training import Trainer
from models.video_model import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from losses.segmentation import BinarySegmentationLoss
from losses.combined import CombinedLoss

def setup_logging(log_dir):
    """Set up logging configuration for training."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Binary Video Segmentation Training')
    parser.add_argument('--config', type=str, default='configs/binary_seg.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments and set up environment
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoints'])
    log_dir = Path(config['paths'].get('logs', 'logs'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting binary video segmentation training with config: {args.config}")
    
    # Set reproducibility seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building binary segmentation model...")
    model = build_model(config).to(device)
    logger.info(f"Model created: {type(model).__name__}")
    
    # Create dataloaders
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        **config['dataset']['augmentation']
    )
    logger.info(f"Created data augmentation with image size: {config['dataset']['img_size']}")
    
    # Create dataloaders
    dataset_params = {
        'batch_size': config['dataset']['batch_size'],
        'img_size': config['dataset']['img_size'],
        'sequence_length': config['dataset']['sequence_length'],
        'sequence_stride': config['dataset']['sequence_stride'],
        'num_workers': config['dataset']['num_workers']
    }
    
    logger.info("Creating train data loader...")
    train_loader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split='train',
        transform=transform,
        **dataset_params
    )
    logger.info(f"Train loader created with {len(train_loader)} batches")
    
    logger.info("Creating validation data loader...")
    val_loader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split='val',
        transform=transform,
        **dataset_params
    )
    logger.info(f"Validation loader created with {len(val_loader)} batches")
    
    # Create optimizer
    logger.info(f"Creating optimizer: {config['optimizer']['type']}")
    optimizer_class = getattr(torch.optim, config['optimizer']['type'])
    optimizer_params = {k: v for k, v in config['optimizer'].items() if k != 'type'}
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # Create learning rate scheduler
    scheduler = None
    step_scheduler_batch = False  # Default value
    
    if 'scheduler' in config and config['scheduler']['type']:
        logger.info(f"Creating scheduler: {config['scheduler']['type']}")
        
        if config['scheduler']['type'] == 'cosine':
            # Convert min_lr to float (handling YAML scientific notation)
            min_lr = float(config['scheduler']['min_lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=min_lr
            )
        elif config['scheduler']['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['scheduler']['step_size'],
                gamma=config['scheduler']['gamma']
            )
        elif config['scheduler']['type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config['scheduler']['factor'],
                patience=config['scheduler']['patience'],
                min_lr=float(config['scheduler']['min_lr'])
            )
        elif config['scheduler']['type'] == 'onecycle':
            from torch.optim.lr_scheduler import OneCycleLR
            
            # Calculate total steps for full training
            total_steps = len(train_loader) * config['training']['epochs']
            
            # Create OneCycleLR scheduler with parameters from config
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['optimizer']['lr'],
                total_steps=total_steps,
                pct_start=config['scheduler'].get('pct_start', 0.1),
                div_factor=config['scheduler'].get('div_factor', 25),
                final_div_factor=config['scheduler'].get('final_div_factor', 1000),
                anneal_strategy='cos'
            )
            step_scheduler_batch = True  # This scheduler should be stepped per batch
    
    # Create trainer with visualization and evaluation capabilities
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        mixed_precision=config['training']['mixed_precision'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        step_scheduler_batch=step_scheduler_batch,  # Pass the flag here
        # Add parameters for visualization and evaluation
        enable_visualization=config.get('visualization', {}).get('enabled', True),
        visualization_dir=config.get('visualization', {}).get('dir', 'visualizations'),
        visualization_interval=config.get('visualization', {}).get('interval', 5),
        enable_evaluation=config.get('evaluation', {}).get('enabled', True)
    )
    logger.info("Trainer initialized")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training process...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        validate_every=config['training']['validate_every'],
        save_every=config['training']['save_every']
    )
    
    # Final evaluation
    logger.info("Training completed. Running final evaluation...")
    final_metrics = trainer.evaluate(
        val_loader=val_loader,
        current_epoch='final',
        visualize=True
    )
    
    # Log final results
    logger.info("Final evaluation results:")
    logger.info(f"IoU: {final_metrics['iou']:.4f}")
    logger.info(f"F1 Score: {final_metrics['f1']:.4f}")
    logger.info(f"Precision: {final_metrics['precision']:.4f}")
    logger.info(f"Recall: {final_metrics['recall']:.4f}")
    logger.info(f"Training and evaluation completed successfully!")
    
    # Save final model configuration and performance
    results_file = checkpoint_dir / 'final_results.yaml'
    with open(results_file, 'w') as f:
        yaml.dump({
            'config': config,
            'final_metrics': {
                k: float(v) if isinstance(v, (float, int)) else v 
                for k, v in final_metrics.items()
            },
            'completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()
```

# utils\__init__.py

```py
from .visualization import VideoSegmentationVisualizer, visualize_instance_tracking
from .evaluation import VideoInstanceEvaluator, DAVISEvaluator

__all__ = [
    'VideoSegmentationVisualizer',
    'visualize_instance_tracking',
    'VideoInstanceEvaluator',
    'DAVISEvaluator'
]
```

# utils\evaluation.py

```py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.optimize
from scipy.ndimage import distance_transform_edt

class VideoInstanceEvaluator:
    """
    Comprehensive evaluation metrics for video instance segmentation.
    Implements J&F measure, temporal consistency, and instance stability metrics.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.scores = {
            'J': [],  # Region similarity (IoU)
            'F': [],  # Boundary similarity
            'T': [],  # Temporal stability
            'instance_stability': []  # Instance ID consistency
        }
    
    def evaluate_sequence(
        self,
        pred_masks: torch.Tensor,  # [T, N, H, W] or [T, 1, H, W]
        gt_masks: torch.Tensor,    # [T, H, W]
        flow: Optional[torch.Tensor] = None  # [T-1, 2, H, W]
    ) -> Dict[str, float]:
        """
        Evaluate a full video sequence.
        
        Args:
            pred_masks: Predicted instance masks [T, N, H, W] or binary mask [T, 1, H, W]
            gt_masks: Ground truth instance masks [T, H, W]
            flow: Optional optical flow between frames [T-1, 2, H, W]
            
        Returns:
            Dictionary with evaluation metrics
        """
        T = pred_masks.shape[0]
        assert gt_masks.shape[0] == T, "Mismatched sequence lengths"
        
        # Get unique IDs in ground truth
        gt_ids = []
        for t in range(T):
            ids = torch.unique(gt_masks[t])
            ids = ids[ids > 0]  # Skip background
            gt_ids.extend(ids.tolist())
        gt_ids = sorted(set(gt_ids))  # Unique IDs across all frames
        
        # Convert predictions to binary instance masks if needed
        if pred_masks.shape[1] == 1:  # Binary segmentation
            binary_preds = (pred_masks > 0.5).squeeze(1)  # [T, H, W]
            pred_instances = [binary_preds] * len(gt_ids)  # Treat as single instance
        else:  # Instance segmentation
            pred_instances = []
            for i in range(min(len(gt_ids), pred_masks.shape[1])):
                pred_instances.append((pred_masks[:, i] > 0.5).squeeze(-1))  # [T, H, W]
        
        # Match predicted instances to ground truth instances
        if len(pred_instances) > 0 and len(gt_ids) > 0:
            matches = self._match_instances(pred_instances, gt_masks, gt_ids)
        else:
            matches = []
        
        # Calculate metrics for each instance
        j_scores = []
        f_scores = []
        
        for pred_idx, gt_id in matches:
            # Get masks for this instance
            pred_masks_inst = pred_instances[pred_idx]  # [T, H, W]
            gt_masks_inst = (gt_masks == gt_id).float()  # [T, H, W]
            
            # Calculate J measure (IoU)
            j_score = self._compute_j_measure(pred_masks_inst, gt_masks_inst)
            j_scores.append(j_score)
            
            # Calculate F measure (boundary similarity)
            f_score = self._compute_f_measure(pred_masks_inst, gt_masks_inst)
            f_scores.append(f_score)
        
        # Calculate temporal stability
        t_score = self._compute_temporal_stability(pred_masks, flow)
        
        # Calculate instance stability
        instance_stability = self._compute_instance_stability(pred_masks)
        
        # Store scores
        self.scores['J'].extend(j_scores)
        self.scores['F'].extend(f_scores)
        self.scores['T'].append(t_score)
        self.scores['instance_stability'].append(instance_stability)
        
        # Calculate mean scores
        mean_j = np.mean(j_scores) if j_scores else 0.0
        mean_f = np.mean(f_scores) if f_scores else 0.0
        
        # Calculate J&F score
        jf_score = (mean_j + mean_f) / 2.0
        
        # Return metrics for this sequence
        return {
            'J&F': jf_score,
            'J_mean': mean_j,
            'F_mean': mean_f,
            'T_mean': t_score,
            'instance_stability': instance_stability
        }
    
    def _match_instances(
        self,
        pred_instances: List[torch.Tensor],  # List of [T, H, W]
        gt_masks: torch.Tensor,             # [T, H, W]
        gt_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Match predicted instances to ground truth instances using Hungarian algorithm.
        
        Args:
            pred_instances: List of predicted instance masks
            gt_masks: Ground truth instance masks
            gt_ids: List of ground truth instance IDs
            
        Returns:
            List of (pred_idx, gt_id) pairs representing matched instances
        """
        num_pred = len(pred_instances)
        num_gt = len(gt_ids)
        
        # If no predictions or no ground truth, return empty list
        if num_pred == 0 or num_gt == 0:
            return []
        
        # Calculate IoU between each prediction and ground truth instance
        iou_matrix = np.zeros((num_pred, num_gt))
        
        for i, pred_mask in enumerate(pred_instances):
            for j, gt_id in enumerate(gt_ids):
                gt_mask = (gt_masks == gt_id).float()
                iou = self._compute_j_measure(pred_mask, gt_mask)
                iou_matrix[i, j] = iou
        
        # Use Hungarian algorithm to find optimal matching
        # We negate IoU because the algorithm minimizes cost
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-iou_matrix)
        
        # Convert to list of (pred_idx, gt_id) pairs
        matches = [(row_ind[i], gt_ids[col_ind[i]]) for i in range(len(row_ind))]
        
        # Filter out matches with IoU below threshold
        iou_threshold = 0.1  # Low threshold to include weak matches
        matches = [(pred_idx, gt_id) for pred_idx, gt_id in matches 
                  if iou_matrix[pred_idx, gt_ids.index(gt_id)] >= iou_threshold]
        
        return matches
    
    def _compute_j_measure(
        self,
        pred_masks: torch.Tensor,  # [T, H, W]
        gt_masks: torch.Tensor     # [T, H, W]
    ) -> float:
        """
        Compute J measure (IoU) for an instance across frames.
        
        Args:
            pred_masks: Predicted masks for one instance
            gt_masks: Ground truth masks for one instance
            
        Returns:
            Mean IoU across frames
        """
        ious = []
        
        for t in range(pred_masks.shape[0]):
            pred = pred_masks[t] > 0.5
            gt = gt_masks[t] > 0.5
            
            # Skip empty frames
            if not gt.any():
                continue
            
            # Calculate IoU
            intersection = (pred & gt).sum().float()
            union = (pred | gt).sum().float()
            
            # Add small epsilon to avoid division by zero
            iou = intersection / (union + 1e-6)
            ious.append(iou.item())
        
        return np.mean(ious) if ious else 0.0
    
    def _compute_f_measure(
        self,
        pred_masks: torch.Tensor,  # [T, H, W]
        gt_masks: torch.Tensor     # [T, H, W]
    ) -> float:
        """
        Compute F measure (boundary similarity) for an instance across frames.
        
        Args:
            pred_masks: Predicted masks for one instance
            gt_masks: Ground truth masks for one instance
            
        Returns:
            Mean F measure across frames
        """
        f_scores = []
        
        for t in range(pred_masks.shape[0]):
            pred = pred_masks[t] > 0.5
            gt = gt_masks[t] > 0.5
            
            # Skip empty frames
            if not gt.any():
                continue
            
            # Calculate precision and recall based on boundary pixels
            pred_boundary = self._get_boundary(pred)
            gt_boundary = self._get_boundary(gt)
            
            # Calculate precision and recall
            precision = (pred_boundary & gt_boundary).sum().float() / (pred_boundary.sum().float() + 1e-6)
            recall = (pred_boundary & gt_boundary).sum().float() / (gt_boundary.sum().float() + 1e-6)
            
            # Calculate F measure
            f_score = (2 * precision * recall) / (precision + recall + 1e-6)
            f_scores.append(f_score.item())
        
        return np.mean(f_scores) if f_scores else 0.0
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Get boundary of a binary mask.
        
        Args:
            mask: Binary mask tensor
            
        Returns:
            Binary boundary mask
        """
        # Convert to numpy for morphological operations
        mask_np = mask.cpu().numpy()
        
        # Calculate distance transform
        dist = distance_transform_edt(mask_np)
        
        # Get boundary pixels (distance = 1)
        boundary = (dist <= 1) & (dist > 0)
        
        # Convert back to tensor
        return torch.from_numpy(boundary).to(mask.device)
    
    def _compute_temporal_stability(
        self,
        pred_masks: torch.Tensor,  # [T, N, H, W] or [T, 1, H, W]
        flow: Optional[torch.Tensor] = None  # [T-1, 2, H, W]
    ) -> float:
        """
        Compute temporal stability score.
        
        Args:
            pred_masks: Predicted instance masks
            flow: Optional optical flow between frames
            
        Returns:
            Temporal stability score
        """
        T = pred_masks.shape[0]
        
        # If only one frame, perfect stability
        if T <= 1:
            return 1.0
        
        # Calculate frame-to-frame changes
        stability_scores = []
        
        for t in range(T - 1):
            curr_mask = pred_masks[t]
            next_mask = pred_masks[t+1]
            
            # If using flow for motion-compensated evaluation
            if flow is not None:
                # Warp current mask to next frame using flow
                curr_flow = flow[t]  # [2, H, W]
                warped_mask = self._warp_mask(curr_mask, curr_flow)
                
                # Calculate stability between warped mask and next mask
                stability = 1.0 - torch.abs(warped_mask - next_mask).mean()
            else:
                # Simple temporal consistency without flow
                stability = 1.0 - torch.abs(curr_mask - next_mask).mean()
            
            stability_scores.append(stability.item())
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _warp_mask(
        self,
        mask: torch.Tensor,  # [N, H, W] or [1, H, W]
        flow: torch.Tensor   # [2, H, W]
    ) -> torch.Tensor:
        """
        Warp mask using optical flow.
        
        Args:
            mask: Mask to warp
            flow: Optical flow field
            
        Returns:
            Warped mask
        """
        # Get mask dimensions
        if mask.dim() == 3:
            N, H, W = mask.shape
        else:
            H, W = mask.shape
            N = 1
            mask = mask.unsqueeze(0)
        
        # Create sampling grid from flow
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=mask.device),
            torch.arange(W, device=mask.device),
            indexing='ij'
        )
        
        # Add flow to grid
        grid_x = grid_x + flow[0]
        grid_y = grid_y + flow[1]
        
        # Normalize grid coordinates to [-1, 1]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        # Stack coordinates
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Reshape mask for grid_sample
        mask_flat = mask.view(1, N, H, W)
        
        # Warp mask using grid sample
        warped_mask = torch.nn.functional.grid_sample(
            mask_flat,
            grid.unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Return warped mask in original shape
        return warped_mask.view(N, H, W)
    
    def _compute_instance_stability(self, pred_masks: torch.Tensor) -> float:
        """
        Compute instance stability score by measuring how consistently
        the model maintains instance identity across frames.
        
        Args:
            pred_masks: Predicted instance masks [T, N, H, W]
            
        Returns:
            Instance stability score
        """
        T, N, H, W = pred_masks.shape
        
        # If only one frame or one instance, perfect stability
        if T <= 1 or N <= 1:
            return 1.0
        
        # Calculate instance consistency across frames
        consistency_matrix = torch.zeros((N, N), device=pred_masks.device)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Calculate temporal IoU between instances i and j
                iou_sequence = []
                
                for t in range(T-1):
                    mask_i_t = pred_masks[t, i] > 0.5
                    mask_j_t1 = pred_masks[t+1, j] > 0.5
                    
                    # Calculate IoU
                    intersection = (mask_i_t & mask_j_t1).sum().float()
                    union = (mask_i_t | mask_j_t1).sum().float()
                    
                    # Add small epsilon to avoid division by zero
                    iou = intersection / (union + 1e-6)
                    iou_sequence.append(iou.item())
                
                # Average IoU between instances i and j across frames
                consistency_matrix[i, j] = np.mean(iou_sequence) if iou_sequence else 0.0
        
        # Calculate stability score
        # Lower IoU between different instances means better separation
        # We take 1 - average IoU between different instances
        stability = 1.0 - consistency_matrix.mean().item()
        
        return stability
    
    def get_global_metrics(self) -> Dict[str, float]:
        """
        Get global metrics averaged across all evaluated sequences.
        
        Returns:
            Dictionary with global metrics
        """
        # Calculate mean scores
        j_mean = np.mean(self.scores['J']) if self.scores['J'] else 0.0
        f_mean = np.mean(self.scores['F']) if self.scores['F'] else 0.0
        t_mean = np.mean(self.scores['T']) if self.scores['T'] else 1.0
        instance_stability = np.mean(self.scores['instance_stability']) if self.scores['instance_stability'] else 1.0
        
        # Calculate J&F score
        jf_score = (j_mean + f_mean) / 2.0
        
        return {
            'J&F': jf_score,
            'J_mean': j_mean,
            'F_mean': f_mean,
            'T_mean': t_mean,
            'instance_stability': instance_stability
        }


class DAVISEvaluator:
    """
    DAVIS benchmark evaluator for video object segmentation.
    Implements the official DAVIS evaluation protocol.
    """
    def __init__(self):
        self.instance_evaluator = VideoInstanceEvaluator()
    
    def evaluate(
        self,
        predictions: List[torch.Tensor],  # List of [T, N, H, W]
        ground_truths: List[torch.Tensor],  # List of [T, H, W]
        sequence_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple sequences using DAVIS protocol.
        
        Args:
            predictions: List of predicted masks for each sequence
            ground_truths: List of ground truth masks for each sequence
            sequence_names: Names of the sequences
            
        Returns:
            Dictionary with per-sequence and global metrics
        """
        # Reset evaluator
        self.instance_evaluator.reset()
        
        # Evaluate each sequence
        sequence_metrics = {}
        
        for pred, gt, name in zip(predictions, ground_truths, sequence_names):
            metrics = self.instance_evaluator.evaluate_sequence(pred, gt)
            sequence_metrics[name] = metrics
        
        # Get global metrics
        global_metrics = self.instance_evaluator.get_global_metrics()
        
        # Return all metrics
        return {
            'global': global_metrics,
            'sequences': sequence_metrics
        }
    # In utils/evaluation.py, update or add this method to DAVISEvaluator

    def evaluate_binary_segmentation(
        self,
        predictions: List[torch.Tensor],  # List of [T, 1, H, W]
        ground_truths: List[torch.Tensor],  # List of [T, H, W]
        sequence_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate binary segmentation performance on DAVIS.
        
        Args:
            predictions: List of predicted binary masks
            ground_truths: List of ground truth masks
            sequence_names: Names of the sequences
            
        Returns:
            Dictionary with evaluation metrics
        """
        sequence_metrics = {}
        global_j_scores = []
        global_f_scores = []
        global_t_scores = []
        
        for pred, gt, name in zip(predictions, ground_truths, sequence_names):
            # Convert predictions to binary
            if pred.dim() == 4:  # [T, 1, H, W]
                pred = pred.squeeze(1)
            binary_pred = (pred > 0.5).bool()
            
            # Convert ground truth to binary
            binary_gt = (gt > 0).bool()
            
            # Compute J measure (IoU)
            j_scores = []
            for t in range(pred.shape[0]):
                intersection = (binary_pred[t] & binary_gt[t]).float().sum()
                union = (binary_pred[t] | binary_gt[t]).float().sum()
                iou = (intersection / (union + 1e-6)).item()
                j_scores.append(iou)
            
            j_mean = np.mean(j_scores)
            global_j_scores.extend(j_scores)
            
            # Compute F measure (boundary precision)
            f_scores = []
            for t in range(pred.shape[0]):
                pred_boundary = self._get_boundary(binary_pred[t])
                gt_boundary = self._get_boundary(binary_gt[t])
                
                precision = (pred_boundary & gt_boundary).float().sum() / (pred_boundary.float().sum() + 1e-6)
                recall = (pred_boundary & gt_boundary).float().sum() / (gt_boundary.float().sum() + 1e-6)
                
                f_score = (2 * precision * recall / (precision + recall + 1e-6)).item()
                f_scores.append(f_score)
            
            f_mean = np.mean(f_scores)
            global_f_scores.extend(f_scores)
            
            # Compute temporal stability
            t_scores = []
            for t in range(pred.shape[0] - 1):
                stability = 1.0 - (binary_pred[t] ^ binary_pred[t+1]).float().mean().item()
                t_scores.append(stability)
            
            t_mean = np.mean(t_scores) if t_scores else 1.0
            global_t_scores.extend(t_scores)
            
            # Store sequence metrics
            sequence_metrics[name] = {
                'J_mean': j_mean,
                'F_mean': f_mean,
                'T_mean': t_mean,
                'J&F': (j_mean + f_mean) / 2
            }
        
        # Compute global metrics
        global_metrics = {
            'J_mean': np.mean(global_j_scores),
            'F_mean': np.mean(global_f_scores),
            'T_mean': np.mean(global_t_scores),
            'J&F': (np.mean(global_j_scores) + np.mean(global_f_scores)) / 2
        }
        
        return {
            'global': global_metrics,
            'sequences': sequence_metrics
        }

    def _get_boundary(self, mask):
        """Helper method to extract boundary pixels from a mask."""
        # Implementation depends on your preference
        dilated = torch.nn.functional.max_pool2d(
            mask.float().unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0) > 0.5
        eroded = torch.nn.functional.avg_pool2d(
            mask.float().unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0) >= 0.9
        
        return dilated & (~eroded)
    
    def print_results(self, results: Dict[str, Dict[str, float]]):
        """
        Print evaluation results in a readable format.
        
        Args:
            results: Dictionary with evaluation results
        """
        global_metrics = results['global']
        sequence_metrics = results['sequences']
        
        print("\n" + "="*50)
        print(f"DAVIS Evaluation Results")
        print("="*50)
        
        # Print global metrics
        print("\nGlobal Metrics:")
        print(f"J&F: {global_metrics['J&F']:.4f}")
        print(f"J-Mean: {global_metrics['J_mean']:.4f}")
        print(f"F-Mean: {global_metrics['F_mean']:.4f}")
        print(f"Temporal Stability: {global_metrics['T_mean']:.4f}")
        print(f"Instance Stability: {global_metrics['instance_stability']:.4f}")
        
        # Print per-sequence metrics
        print("\nPer-Sequence Metrics:")
        for name, metrics in sequence_metrics.items():
            print(f"\n{name}:")
            print(f"  J&F: {metrics['J&F']:.4f}")
            print(f"  J-Mean: {metrics['J_mean']:.4f}")
            print(f"  F-Mean: {metrics['F_mean']:.4f}")
            print(f"  Temporal Stability: {metrics['T_mean']:.4f}")
            print(f"  Instance Stability: {metrics['instance_stability']:.4f}")
        
        print("\n" + "="*50)
```

# utils\training.py

```py


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
from tqdm import tqdm

from losses.combined import CombinedLoss
from losses.temporal_consistency import TemporalConsistencyLoss

from losses.segmentation import BinarySegmentationLoss

import torch
import torch.nn as nn
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Union
from models.backbone import TemporalFeatureBank


# At the top of your file, outside any class
def get_item_safely(value):
    """Safely extract item from tensor or return float value."""
    if hasattr(value, 'item'):
        return value.item()
    return value

class Trainer:
    """Handles the complete training process including checkpointing and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        step_scheduler_batch: bool = False,  # Add this parameter
        # Add new parameters for visualization and evaluation
        enable_visualization: bool = True,
        visualization_dir: str = 'visualizations',
        visualization_interval: int = 5,
        enable_evaluation: bool = True
    ):
        # Existing initialization code
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Replace the current criterion with binary segmentation loss
        # Initialize the loss function with the weights provided in the constructor
        self.criterion = BinarySegmentationLoss(
            ce_weight=config['losses']['ce_weight'],
            dice_weight=config['losses']['dice_weight'],
            boundary_weight=config['losses'].get('boundary_weight', 0.0)  # Add default for backward compatibility
        )
        # Modern mixed precision setup
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_metrics = {}

        self.grad_clip_value = config['training'].get('grad_clip_value', 0.0)
        self.step_scheduler_batch = step_scheduler_batch  # Use the parameter value
    
    # Rest of your initialization code...
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize visualization and evaluation tools if enabled
        self.enable_visualization = enable_visualization
        self.visualization_interval = visualization_interval
        if enable_visualization:
            from utils.visualization import VideoSegmentationVisualizer
            self.visualization_dir = Path(visualization_dir)
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            self.visualizer = VideoSegmentationVisualizer(save_dir=self.visualization_dir)
        
        self.enable_evaluation = enable_evaluation
        if enable_evaluation:
            from utils.evaluation import DAVISEvaluator
            self.evaluator = DAVISEvaluator()

    
    def get_current_lr(self):
        """Get the current learning rate from the optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def current_epoch(self):
        """Get the current training epoch."""
        return getattr(self, '_current_epoch', 0)
        
    @current_epoch.setter
    def current_epoch(self, epoch):
        """Set the current training epoch."""
        self._current_epoch = epoch
    
    def save_checkpoint(self, metrics: Dict[str, float], name: str = 'model') -> None:
        """Saves a checkpoint of the current training state."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        save_path = self.checkpoint_dir / f'{name}.pth'
        torch.save(checkpoint, save_path)
        
        # Also save metrics separately for easy access
        metrics_path = self.checkpoint_dir / f'{name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved checkpoint and metrics to {self.checkpoint_dir}")
    
    def load_checkpoint(self, path: str, load_best: bool = True) -> None:
        """Loads a checkpoint and restores the training state."""
        path = Path(path)
        if load_best:
            path = path.parent / f'{path.stem}_best.pth'
        
        if not path.exists():
            self.logger.warning(f"No checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        # Restore scheduler if it exists
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Restored checkpoint from {path} (epoch {self.epoch})")
    
    
    def train_epoch(self, train_loader):
        """
        Run a single training epoch with binary segmentation handling.
        
        This method processes each batch of video data, computes the loss,
        performs backpropagation, and tracks training metrics throughout the epoch.
        
        Args:
            train_loader: DataLoader providing training batches
            
        Returns:
            Dictionary containing average loss values and metrics for the epoch
        """
        self.model.train()
        
        # Initialize tracking variables
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        total_boundary_loss = 0.0  # Track boundary loss
        
        # Progress metrics
        batch_count = len(train_loader)
        processed_samples = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=batch_count, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
                masks = batch['masks'].to(self.device)    # [B, T, H, W]
                
                # Track batch size for averaging
                batch_size = frames.shape[0]
                processed_samples += batch_size
                
                # Mixed precision training if enabled
                with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                    # Forward pass - model outputs dict with 'logits' and 'pred_masks'
                    outputs = self.model(frames)
                    
                    # Compute loss - expects dict with 'masks' key
                    loss_dict = self.criterion(outputs, {'masks': masks})
                    
                    # Get individual loss components
                    loss = loss_dict['loss']  # Total loss
                    ce_loss = loss_dict.get('ce_loss', 0.0)
                    dice_loss = loss_dict.get('dice_loss', 0.0)
                    boundary_loss = loss_dict.get('boundary_loss', 0.0)  # Get boundary loss
                
                # Backward pass with gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    # Only update weights after accumulating enough gradients
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Unscale gradients for clipping (if used)
                        if self.grad_clip_value > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                        
                        # Optimizer step with scaler for mixed precision
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                        
                        # Step the scheduler if it's batch-based
                        if self.scheduler is not None and self.step_scheduler_batch:
                            self.scheduler.step()
                else:
                    # Standard backward and update (no accumulation)
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping if configured
                    if self.grad_clip_value > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Step the scheduler if it's batch-based
                    if self.scheduler is not None and self.step_scheduler_batch:
                        self.scheduler.step()
                
                # Update tracking metrics
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss * batch_size if isinstance(ce_loss, float) else ce_loss.item() * batch_size
                total_dice_loss += dice_loss * batch_size if isinstance(dice_loss, float) else dice_loss.item() * batch_size
                
                # Update boundary loss tracking if present
                if boundary_loss != 0:
                    total_boundary_loss += boundary_loss * batch_size if isinstance(boundary_loss, float) else boundary_loss.item() * batch_size
                
                # Update progress bar with current loss values
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'ce': f"{ce_loss if isinstance(ce_loss, float) else ce_loss.item():.4f}",
                    'dice': f"{dice_loss if isinstance(dice_loss, float) else dice_loss.item():.4f}",
                    'lr': f"{self.get_current_lr():.6f}"
                }
                
                # Add boundary loss to progress bar if available
                if boundary_loss != 0:
                    postfix_dict['bound'] = f"{boundary_loss if isinstance(boundary_loss, float) else boundary_loss.item():.4f}"
                    
                pbar.update(1)
                pbar.set_postfix(postfix_dict)
                
                # Optional logging for step-wise metrics (e.g., TensorBoard)
                if hasattr(self, 'log_metrics') and callable(getattr(self, 'log_metrics')):
                    step = batch_idx + (self.current_epoch * batch_count)
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/ce_loss': ce_loss if isinstance(ce_loss, float) else ce_loss.item(),
                        'train/dice_loss': dice_loss if isinstance(dice_loss, float) else dice_loss.item(),
                        'train/lr': self.get_current_lr()
                    }
                    
                    # Add boundary loss to logging if available
                    if boundary_loss != 0:
                        log_dict['train/boundary_loss'] = boundary_loss if isinstance(boundary_loss, float) else boundary_loss.item()
                        
                    self.log_metrics(log_dict, step)
            
            # Compute average metrics for the epoch
            avg_loss = total_loss / processed_samples
            avg_ce_loss = total_ce_loss / processed_samples
            avg_dice_loss = total_dice_loss / processed_samples
            avg_boundary_loss = total_boundary_loss / processed_samples if total_boundary_loss > 0 else 0.0
            
            # Step the scheduler if it's epoch-based
            if self.scheduler is not None and not self.step_scheduler_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch completion
            log_message = (
                f"Epoch {self.current_epoch} completed: "
                f"Loss: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f}"
            )
            
            # Add boundary loss to log if non-zero
            if avg_boundary_loss > 0:
                log_message += f", Boundary: {avg_boundary_loss:.4f}"
            
            log_message += f", LR: {self.get_current_lr():.6f}"
            self.logger.info(log_message)
            
            # Return average metrics
            result = {
                'loss': avg_loss,
                'ce_loss': avg_ce_loss,
                'dice_loss': avg_dice_loss
            }
            
            # Add boundary loss to results if non-zero
            if avg_boundary_loss > 0:
                result['boundary_loss'] = avg_boundary_loss
            
            return result
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation with memory optimizations."""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        # For evaluation
        all_predictions = []
        all_ground_truths = []
        sequence_names = []
        
        # Process in smaller chunks to avoid memory issues
        max_sequences_per_chunk = 20  # Adjust based on your system's RAM
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for batch_idx, batch in enumerate(pbar):
                try:
                    frames = batch['frames'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    sequence_name = batch['sequence'][0] if 'sequence' in batch else f"sequence_{batch_idx}"
                    
                    with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                        outputs = self.model(frames)
                        losses = self.criterion(outputs, {'masks': masks})
                        total_loss += losses['loss'].item()  # Use 'loss' instead of 'total_loss'
                    
                    # Store predictions for evaluation (but limit memory usage)
                    if len(all_predictions) < max_sequences_per_chunk:
                        all_predictions.append(outputs['pred_masks'][0].cpu())
                        all_ground_truths.append(masks[0].cpu())
                        sequence_names.append(sequence_name)
                    
                    # Visualize only occasionally
                    if self.enable_visualization and self.global_step % (self.visualization_interval * 10) == 0 and batch_idx % 50 == 0:
                        self.visualizer.visualize_sequence(
                            frames=frames[0].cpu(),
                            pred_masks=outputs['pred_masks'][0].cpu(),
                            gt_masks=masks[0].cpu(),
                            sequence_name=f"{sequence_name}_epoch_{self.epoch}_lite",
                            max_frames=2  # Only visualize 2 frames, not the whole sequence
                        )
                    
                    # Clear memory
                    del outputs, frames, masks
                    torch.cuda.empty_cache()
                    
                    # Evaluate and reset if we've accumulated enough sequences
                    if len(all_predictions) >= max_sequences_per_chunk or batch_idx == num_batches - 1:
                        if all_predictions and self.enable_evaluation:
                            eval_metrics = self._evaluate_current_predictions(all_predictions, all_ground_truths, sequence_names)
                            
                            # Clear predictions to free memory
                            all_predictions = []
                            all_ground_truths = []
                            sequence_names = []
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.warning(f"OOM during validation at batch {batch_idx}. Clearing memory and continuing...")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        
                        # Clear accumulated data
                        all_predictions = []
                        all_ground_truths = []
                        sequence_names = []
                        
                        continue
                    else:
                        raise e
        
        # Calculate metrics
        metrics = {'val_loss': total_loss / num_batches}
        
        # Add evaluation metrics if available
        if hasattr(self, 'eval_metrics') and self.eval_metrics:
            for key, value in self.eval_metrics.items():
                metrics[key] = value
        
        return metrics

    def _evaluate_current_predictions(self, predictions, ground_truths, seq_names):
        """Helper method to evaluate accumulated predictions in chunks."""
        if self.enable_evaluation and predictions:
            try:
                eval_results = self.evaluator.evaluate(
                    predictions=predictions,
                    ground_truths=ground_truths,
                    sequence_names=seq_names
                )
                
                # Store global metrics for reporting
                self.eval_metrics = eval_results['global']
                
                # Only log abbreviated results
                self.logger.info("\nPartial Evaluation Results:")
                self.logger.info(f"Global: J&F: {eval_results['global']['J&F']:.4f}, "
                                f"J_mean: {eval_results['global']['J_mean']:.4f}, "
                                f"T_mean: {eval_results['global']['T_mean']:.4f}")
                
                return eval_results['global']
                
            except Exception as e:
                self.logger.error(f"Error during evaluation: {str(e)}")
                return {}
        return {}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        validate_every: int = 1,
        save_every: int = 10
    ):
        """
        Main training loop with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Total number of epochs to train
            validate_every: Frequency of validation in epochs
            save_every: Frequency of checkpointing in epochs
        """
        self.logger.info(f"Starting training from epoch {self.epoch}")
        
        for epoch in range(self.epoch, num_epochs):
            for module in self.model.modules():
                if isinstance(module, TemporalFeatureBank):
                    module.features.clear()
            self.epoch = epoch
            self.current_epoch = epoch  # Add this line to update the property
            
            # Training
            train_loss = self.train_epoch(train_loader)
            #self.logger.info(f"Epoch {epoch} training loss: {train_loss['loss']:.4f}")

            
            # Validation
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_losses = self.validate(val_loader)
                self.logger.info(
                    f"Epoch {epoch} validation: " +
                    " ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
                )
                
                # Save checkpoint if it's the best model so far
                if val_losses['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['val_loss']
                    self.save_checkpoint(val_losses, name='model_best')
                
                # Regular checkpoint saving
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(val_losses, name=f'model_epoch_{epoch}')
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.logger.info("Training completed!")

    def compute_binary_metrics(predictions, ground_truths):
        """
        Compute metrics for binary video segmentation evaluation.
        
        This function calculates IoU (Intersection over Union), precision, recall,
        and F1 score for binary segmentation masks. Each metric is calculated per frame
        and then averaged across all frames in all sequences.
        
        Args:
            predictions: List of prediction tensors, each with shape [T, 1, H, W]
            ground_truths: List of ground truth tensors, each with shape [T, H, W]
            
        Returns:
            Dictionary containing averaged metrics:
            - iou: Intersection over Union (Jaccard index)
            - precision: Precision (TP / (TP + FP))
            - recall: Recall (TP / (TP + FN))
            - f1: F1 score (2 * precision * recall / (precision + recall))
        """
        # Initialize accumulators
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_frames = 0
        
        # Process each sequence
        for pred, gt in zip(predictions, ground_truths):
            # Convert predictions to binary (threshold at 0.5)
            # Ensure we're working with the correct dimensions
            if pred.dim() == 4:  # [T, 1, H, W]
                binary_pred = (pred.squeeze(1) > 0.5).bool()
            else:  # [T, H, W]
                binary_pred = (pred > 0.5).bool()
            
            # Convert ground truth to binary (values > 0 are foreground)
            binary_gt = (gt > 0).bool()
            
            # Process each frame in the sequence
            for t in range(binary_pred.shape[0]):
                # Calculate true positives, false positives, false negatives
                tp = (binary_pred[t] & binary_gt[t]).sum().float()
                fp = (binary_pred[t] & ~binary_gt[t]).sum().float()
                fn = (~binary_pred[t] & binary_gt[t]).sum().float()
                
                # Calculate metrics (add small epsilon to avoid division by zero)
                epsilon = 1e-8
                intersection = tp
                union = tp + fp + fn
                
                iou = intersection / (union + epsilon)
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2 * precision * recall / (precision + recall + epsilon)
                
                # Accumulate metrics
                total_iou += iou.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item()
                total_frames += 1
                
                # Log individual frame metrics for debugging (optional)
                # logger.debug(f"Frame {total_frames}: IoU={iou.item():.4f}, F1={f1.item():.4f}")
        
        # Handle empty case
        if total_frames == 0:
            logger.warning("No frames were processed during metric calculation")
            return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate and return averages
        return {
            'iou': total_iou / total_frames,
            'precision': total_precision / total_frames,
            'recall': total_recall / total_frames,
            'f1': total_f1 / total_frames
        }
    def evaluate(self, val_loader, visualize=False):
        """
        Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Import visualizer if visualization is requested
        if visualize:
            from utils.visualization import BinarySegmentationVisualizer
            visualizer = BinarySegmentationVisualizer(
                save_dir=os.path.join(self.checkpoint_dir, 'visualizations')
            )
        
        self.model.eval()
        
        # Initialize tracking variables
        total_loss = 0.0
        batch_sizes = 0
        all_predictions = []
        all_ground_truths = []
        all_frames = []
        all_seq_names = []
        
        # Evaluate without gradient computation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get data from batch
                frames = batch['frames'].to(self.device)
                masks = batch['masks'].to(self.device)
                sequence_names = batch.get('sequence_name', [f"seq_{i}" for i in range(frames.shape[0])])
                
                # Forward pass
                outputs = self.model(frames)
                
                # Compute loss
                loss_dict = self.criterion(outputs, {'masks': masks})
                loss = loss_dict.get('loss', loss_dict.get('total_loss', 0.0))
                
                # Track metrics
                batch_size = frames.shape[0]
                total_loss += loss.item() * batch_size
                batch_sizes += batch_size
                
                # Store predictions and ground truth for metrics
                pred_masks = outputs['pred_masks']  # [B, T, 1, H, W]
                
                # Store results for each sequence
                for i in range(frames.shape[0]):
                    all_predictions.append(pred_masks[i])
                    all_ground_truths.append(masks[i])
                    all_frames.append(frames[i])
                    all_seq_names.append(sequence_names[i])
        
        # Calculate average loss
        avg_loss = total_loss / batch_sizes
        
        # Compute segmentation metrics
        metrics = self.compute_binary_metrics(all_predictions, all_ground_truths)
        metrics['loss'] = avg_loss
        
        # Generate visualizations if requested
        if visualize:
            num_viz = min(5, len(all_frames))
            for i in range(num_viz):
                viz_path = os.path.join(
                    visualizer.save_dir, 
                    f"{all_seq_names[i]}_epoch{getattr(self, 'current_epoch', 0)}.png"
                )
                visualizer.visualize_predictions(
                    all_frames[i].unsqueeze(0),
                    all_predictions[i].unsqueeze(0),
                    all_ground_truths[i].unsqueeze(0),
                    save_path=viz_path
                )
        
        # Log metrics
        self.logger.info(
            f"Validation - Loss: {avg_loss:.4f}, "
            f"IoU: {metrics['iou']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )
        
        return metrics
            
```

# utils\visualization.py

```py
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
```

# visualizations\rtx3070\bike-packing_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bike-packing_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\blackswan_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\bmx-trees_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\breakdance_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\camel_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-roundabout_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\car-shadow_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\cows_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dance-twirl_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dog_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\dogs-jump_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-chicane_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\drift-straight_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_0_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_1_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_2_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_3_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_4_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_5_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\evaluation_epoch_6_dashboard.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\goat_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\gold-fish_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\horsejump-high_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\india_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\judo_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\kite-surf_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\lab-coat_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\libby_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\loading_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\mbike-trick_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\motocross-jump_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\paragliding-launch_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\parkour_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\pigs_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\scooter-black_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\shooting_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_1_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_1_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_2_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_2_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_3_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_3_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_4_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_4_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_5_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_5_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_6_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx3070\soapbox_epoch_6_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_1_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_1_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_3_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_3_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_5_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_5_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_7_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_7_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_9_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_9_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_11_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_11_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_13_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_13_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_15_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_15_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_17_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_17_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_19_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_19_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_21_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_21_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_23_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_23_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_25_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_25_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_27_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_27_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_29_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_29_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_31_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_31_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_33_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_33_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_35_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_35_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_37_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_37_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_39_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_39_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_41_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_41_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_43_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_43_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_45_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_45_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_47_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\bike-packing_epoch_47_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_1_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_1_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_3_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_3_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_5_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_5_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_7_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_7_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_9_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_9_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_11_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_11_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_13_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_13_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_15_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_15_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_17_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_17_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_19_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_19_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_21_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_21_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_23_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_23_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_25_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_25_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_27_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_27_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_29_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_29_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_31_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_31_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_33_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_33_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_35_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_35_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_37_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_37_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_39_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_39_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_41_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_41_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_43_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_43_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_45_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_45_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_47_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\car-roundabout_epoch_47_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_1_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_1_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_3_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_3_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_5_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_5_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_7_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_7_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_9_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_9_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_11_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_11_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_13_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_13_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_15_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_15_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_17_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_17_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_19_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_19_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_21_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_21_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_23_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_23_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_25_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_25_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_27_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_27_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_29_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_29_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_31_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_31_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_33_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_33_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_35_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_35_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_37_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_37_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_39_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_39_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_41_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_41_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_43_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_43_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_45_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_45_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_47_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\dogs-jump_epoch_47_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_1_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_1_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_3_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_3_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_5_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_5_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_7_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_7_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_9_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_9_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_11_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_11_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_13_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_13_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_15_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_15_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_17_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_17_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_19_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_19_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_21_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_21_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_23_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_23_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_25_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_25_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_27_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_27_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_29_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_29_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_31_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_31_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_33_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_33_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_35_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_35_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_37_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_37_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_39_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_39_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_41_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_41_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_43_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_43_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_45_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_45_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_47_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\india_epoch_47_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_1_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_1_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_3_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_3_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_5_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_5_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_7_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_7_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_9_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_9_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_11_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_11_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_13_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_13_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_15_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_15_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_17_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_17_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_19_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_19_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_21_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_21_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_23_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_23_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_25_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_25_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_27_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_27_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_29_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_29_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_31_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_31_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_33_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_33_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_35_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_35_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_37_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_37_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_39_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_39_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_41_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_41_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_43_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_43_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_45_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_45_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_47_lite_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti_super_fast\paragliding-launch_epoch_47_lite_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\bike-packing_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\bike-packing_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\bike-packing_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\blackswan_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\blackswan_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\blackswan_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\bmx-trees_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\bmx-trees_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\bmx-trees_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\breakdance_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\breakdance_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\breakdance_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\camel_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\camel_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\camel_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\car-roundabout_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\car-roundabout_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\car-roundabout_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\car-shadow_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\car-shadow_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\car-shadow_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\cows_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\cows_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\cows_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dance-twirl_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dance-twirl_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dance-twirl_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dog_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dog_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dog_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dogs-jump_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dogs-jump_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\dogs-jump_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\drift-chicane_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\drift-chicane_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\drift-chicane_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\drift-straight_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\drift-straight_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\drift-straight_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\goat_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\goat_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\goat_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\gold-fish_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\gold-fish_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\gold-fish_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\horsejump-high_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\horsejump-high_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\horsejump-high_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\india_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\india_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\india_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\judo_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\judo_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\judo_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\kite-surf_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\kite-surf_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\kite-surf_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\lab-coat_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\lab-coat_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\lab-coat_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\libby_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\libby_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\libby_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\loading_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\loading_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\loading_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\mbike-trick_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\mbike-trick_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\mbike-trick_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\motocross-jump_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\motocross-jump_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\motocross-jump_epoch_0_frame_002.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\paragliding-launch_epoch_0_frame_000.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\paragliding-launch_epoch_0_frame_001.png

This is a binary file of the type: Image

# visualizations\rtx4070ti\paragliding-launch_epoch_0_frame_002.png

This is a binary file of the type: Image

