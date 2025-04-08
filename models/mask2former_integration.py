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