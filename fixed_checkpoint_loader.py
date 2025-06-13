
import torch
from collections import OrderedDict

def load_model_with_checkpoint_fix(model, checkpoint_path, device='cuda'):
    """Load model with automatic error handling."""
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try loading with strict=False to ignore mismatched keys
        result = model.load_state_dict(state_dict, strict=False)
        
        print(f"✅ Checkpoint loaded successfully")
        if result.missing_keys:
            print(f"⚠️  Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(result.unexpected_keys)}")
            
        return model, True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Continuing with untrained model...")
        return model, False
