import os
import torch
import yaml
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
import json

# Import project components
from models.binary_mamba_segmentation import build_model
from datasets.davis import build_davis_dataloader
from datasets.transforms import VideoSequenceAugmentation
from utils.training import Trainer

import math


def inspect_model(model):
    """Print model details and parameter distributions to help diagnose issues."""
    print("\nModel Architecture Overview:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check for problematic parameter initialization
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            if std < 1e-8 or math.isnan(std) or math.isnan(mean):
                print(f"WARNING - Problem with {name}: mean={mean:.6f}, std={std:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
    
    # Check for non-changing parameters after a forward pass
    x = torch.randn(2, 3, 3, 240, 320).to(next(model.parameters()).device)
    before = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            before[name] = param.data.clone()
    
    # Forward pass
    with torch.enable_grad():
        outputs = model(x)
        loss = outputs['logits'].abs().mean()
        loss.backward()
    
    # Check which parameters didn't get gradients
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            no_grad_params.append(name)
    
    if no_grad_params:
        print("\nParameters without gradients:")
        for name in no_grad_params:
            print(f"- {name}")
    
    return model

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
    parser = argparse.ArgumentParser(description='Mamba-based Video Segmentation Training')
    parser.add_argument('--config', type=str, default='configs/mamba_binary_efficient.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--find-lr', action='store_true', help='Run learning rate finder before training')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller dataset')
    return parser.parse_args()

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def test_model_forward(model, device):
    """Test the model's forward pass with a small batch."""
    # Create dummy input
    batch = 2
    frames = 3
    channels = 3
    height = 128
    width = 128
    x = torch.randn(batch, frames, channels, height, width).to(device)
    
    # Run forward pass
    with torch.inference_mode():
        outputs = model(x)
    
    # Check output shape
    print("\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Output masks shape: {outputs['pred_masks'].shape}")
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return outputs

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set up directories and logging
    checkpoint_dir = Path(config['paths']['checkpoints'])
    log_dir = Path(config['paths'].get('logs', 'logs'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_dir)
    logger.info(f"Starting Mamba-based binary video segmentation training")
    logger.info(f"Using configuration from: {args.config}")
    
    # Save config to checkpoint directory for reference
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Set reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building Mamba-based binary segmentation model...")
    model = build_model(config).to(device)
    model = inspect_model(model)  # Add this line
    # Build model

    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} trainable parameters")
    
    # Test model forward pass
    if not args.resume:
        test_model_forward(model, device)
    
    # Create data transforms
    transform = VideoSequenceAugmentation(
        img_size=tuple(config['dataset']['img_size']),
        **config['dataset']['augmentation']
    )
    
    # Create dataloaders
    logger.info("Creating data loaders...")
    
    if args.debug:
        # Use smaller dataset for debugging
        config['dataset']['sequence_length'] = min(config['dataset']['sequence_length'], 2)
        config['dataset']['batch_size'] = 1
    
    train_loader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split='train',
        transform=transform,
        **{k: v for k, v in config['dataset'].items() 
           if k not in ['augmentation']}
    )
    
    val_loader = build_davis_dataloader(
        root_path=config['paths']['davis_root'],
        split='val',
        transform=VideoSequenceAugmentation(
            img_size=tuple(config['dataset']['img_size']),
            train=False
        ),
        **{k: v for k, v in config['dataset'].items() 
           if k not in ['augmentation']}
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Create optimizer
    logger.info(f"Creating optimizer: {config['optimizer']['type']}")
    optimizer_class = getattr(torch.optim, config['optimizer']['type'])
    optimizer_params = {k: v for k, v in config['optimizer'].items() 
                       if k != 'type'}
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # Create scheduler
    scheduler = None
    step_scheduler_batch = False
    
    if config['scheduler']['type'] == 'onecycle':
        # Calculate total steps
        total_steps = len(train_loader) * config['training']['epochs']
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optimizer']['lr'],
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000
        )
        step_scheduler_batch = True
        logger.info(f"Created OneCycleLR scheduler with max_lr={config['optimizer']['lr']}")
    elif config['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=float(config['scheduler']['min_lr'])
        )
        logger.info(f"Created CosineAnnealingLR scheduler")
    elif config['scheduler']['type'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=float(config['scheduler']['min_lr'])
        )
        logger.info(f"Created ReduceLROnPlateau scheduler")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        mixed_precision=config['training']['mixed_precision'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        step_scheduler_batch=step_scheduler_batch,
        enable_visualization=config.get('visualization', {}).get('enabled', True),
        visualization_dir=config.get('visualization', {}).get('dir', 'visualizations'),
        visualization_interval=config.get('visualization', {}).get('interval', 5),
        enable_evaluation=config.get('evaluation', {}).get('enabled', True)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Optionally find optimal learning rate
    if args.find_lr:
        logger.info("Running learning rate finder...")
        suggested_lr = trainer.find_learning_rate(
            train_loader=train_loader,
            start_lr=1e-6,
            end_lr=1e-2,
            num_iterations=50
        )
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = suggested_lr
        
        logger.info(f"Updated learning rate to {suggested_lr:.1e}")
    
    # Train model
    logger.info("Starting training loop...")
    start_time = time.time()
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        validate_every=config['training']['validate_every'],
        save_every=config['training']['save_every'],
        patience=config.get('training', {}).get('patience', 15)  # Early stopping patience
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time / 3600:.2f} hours")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate(val_loader, visualize=True)
    
    # Log final metrics
    logger.info("Final evaluation results:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save final results
    with open(checkpoint_dir / 'final_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in final_metrics.items()}, f, indent=2)
        
    logger.info(f"All results saved to {checkpoint_dir}")
    
    # Print final summary
    logger.info("\n" + "="*50)
    logger.info("Training Summary:")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Total training time: {training_time / 3600:.2f} hours")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final metrics: J&F={final_metrics.get('J&F', 0):.4f}, IoU={final_metrics.get('J_mean', 0):.4f}")
    logger.info("="*50)

if __name__ == '__main__':
    main()