# __init__.py

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

# backbone.py

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
            # Resize historical features to match current feature size if needed
            if hist_features.shape[2:] != current_features.shape[2:]:
                hist_features = F.interpolate(
                    hist_features, 
                    size=(H, W),
                    mode='bilinear', 
                    align_corners=False
                )
                # Also resize mask - convert to float for interpolation
                if mask.shape[2:] != current_features.shape[2:]:
                    mask_float = mask.float()  # Convert bool to float
                    mask_float = F.interpolate(
                        mask_float, 
                        size=(H, W),
                        mode='nearest'
                    )
                    mask = mask_float > 0.5  # Convert back to bool
                    
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

# decoder.py

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

# mask2former_integration.py

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

# model.py

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

# position_encoding.py

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

# temporal_components.py

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

# video_instance_decoder.py

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

# video_model.py

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
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Process through binary segmentation head
        logits = self.seg_head(backbone_features)
        
        # Reshape to add temporal dimension
        logits = logits.view(B, T, 1, H, W)  # Note: Only 1 channel now
        
        # Apply temporal smoothing
        smoothed_logits = self.temporal_smooth(logits)
        
        # Return binary segmentation mask
        return {
            'pred_masks': smoothed_logits,
            'logits': logits  # For loss computation
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

