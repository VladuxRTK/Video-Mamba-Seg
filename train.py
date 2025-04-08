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