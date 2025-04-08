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