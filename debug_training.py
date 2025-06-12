#!/usr/bin/env python3
"""
Emergency debug script to identify why training is failing.
Run this before starting new training to validate everything works.
"""

import torch
import yaml
import numpy as np
from pathlib import Path

def debug_model_and_loss():
    """Debug the model and loss computation."""
    print("🔍 DEBUGGING MODEL AND LOSS")
    print("=" * 50)
    
    # Load config
    config_path = "configs/emergency_fix.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded from {config_path}")
    
    # Test model creation
    try:
        from models.binary_mamba_segmentation import build_model
        model = build_model(config)
        print(f"✅ Model created successfully")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    B, T, C, H, W = 1, 3, 3, 240, 320
    x = torch.randn(B, T, C, H, W).to(device)
    masks = torch.randint(0, 2, (B, T, H, W)).float().to(device)
    
    print(f"✅ Created dummy data: {x.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(x)
        print(f"✅ Forward pass successful")
        print(f"   - Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"   - {key}: {value.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test loss computation
    try:
        from losses.combined import EnhancedCombinedLoss
        criterion = EnhancedCombinedLoss(
            ce_weight=config['losses']['ce_weight'],
            dice_weight=config['losses']['dice_weight'],
            temporal_weight=config['losses']['temporal_weight'],
            boundary_weight=config['losses'].get('boundary_weight', 0.0)
        )
        
        loss_dict = criterion(outputs, {'masks': masks})
        print(f"✅ Loss computation successful")
        print(f"   - Total loss: {loss_dict['loss'].item():.6f}")
        for key, value in loss_dict.items():
            if key != 'loss':
                val = value.item() if hasattr(value, 'item') else value
                print(f"   - {key}: {val:.6f}")
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # Test backward pass
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        outputs = model(x)
        loss_dict = criterion(outputs, {'masks': masks})
        loss = loss_dict['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Backward pass successful")
        print(f"   - Loss before: {loss.item():.6f}")
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        print(f"   - Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    return True

def debug_data_loading():
    """Debug data loading and preprocessing."""
    print("\n🔍 DEBUGGING DATA LOADING")
    print("=" * 50)
    
    try:
        from datasets.davis import build_davis_dataloader
        from datasets.transforms import VideoSequenceAugmentation
        
        # Load config
        with open("configs/emergency_fix.yaml") as f:
            config = yaml.safe_load(f)
        
        # Create transform
        transform = VideoSequenceAugmentation(
            img_size=tuple(config['dataset']['img_size']),
            **config['dataset']['augmentation'],
            train=True
        )
        
        # Create dataloader
        train_loader = build_davis_dataloader(
            root_path=config['paths']['davis_root'],
            split='train',
            transform=transform,
            batch_size=1,  # Use batch size 1 for testing
            **{k: v for k, v in config['dataset'].items() 
               if k not in ['augmentation', 'batch_size']}
        )
        
        print(f"✅ DataLoader created: {len(train_loader)} batches")
        
        # Test loading one batch
        batch = next(iter(train_loader))
        print(f"✅ Batch loaded successfully")
        print(f"   - Frames shape: {batch['frames'].shape}")
        print(f"   - Masks shape: {batch['masks'].shape}")
        print(f"   - Sequence: {batch['sequence']}")
        
        # Check data ranges
        frames = batch['frames']
        masks = batch['masks']
        
        print(f"   - Frames range: [{frames.min():.3f}, {frames.max():.3f}]")
        print(f"   - Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
        print(f"   - Masks unique values: {torch.unique(masks)}")
        
        # Check foreground ratio
        fg_ratio = (masks > 0).float().mean().item()
        print(f"   - Foreground ratio: {fg_ratio:.4f}")
        
        if fg_ratio < 0.01:
            print("   ⚠️  WARNING: Very low foreground ratio - check your masks!")
        elif fg_ratio > 0.8:
            print("   ⚠️  WARNING: Very high foreground ratio - check your masks!")
        else:
            print("   ✅ Foreground ratio looks reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def debug_training_setup():
    """Debug the training setup."""
    print("\n🔍 DEBUGGING TRAINING SETUP")
    print("=" * 50)
    
    # Check paths
    with open("configs/emergency_fix.yaml") as f:
        config = yaml.safe_load(f)
    
    davis_path = Path(config['paths']['davis_root'])
    if not davis_path.exists():
        print(f"❌ DAVIS dataset not found at: {davis_path}")
        return False
    else:
        print(f"✅ DAVIS dataset found at: {davis_path}")
    
    # Check required subdirectories
    required_dirs = ['JPEGImages/480p', 'Annotations/480p', 'ImageSets/2017']
    for subdir in required_dirs:
        full_path = davis_path / subdir
        if not full_path.exists():
            print(f"❌ Missing directory: {full_path}")
        else:
            print(f"✅ Found directory: {full_path}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available - using CPU")
    
    return True

def main():
    """Run all debug checks."""
    print("🚀 EMERGENCY TRAINING DEBUG")
    print("=" * 60)
    
    all_passed = True
    
    # Run all debug checks
    all_passed &= debug_training_setup()
    all_passed &= debug_data_loading()
    all_passed &= debug_model_and_loss()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Ready to start training.")
        print("\nRun training with:")
        print("python train.py --config configs/emergency_fix.yaml")
    else:
        print("❌ SOME CHECKS FAILED! Fix the issues before training.")
    print("=" * 60)

if __name__ == "__main__":
    main()