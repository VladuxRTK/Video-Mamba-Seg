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