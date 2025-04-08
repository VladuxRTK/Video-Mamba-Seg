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