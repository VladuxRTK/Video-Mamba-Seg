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