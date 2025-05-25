import torch
import torch.nn as nn
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import cv2
import math
import time
import csv
import datetime
import copy
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from pathlib import Path
from torch.utils.data import DataLoader

# Helper function to safely extract item from tensor
def get_item_safely(value):
    """Safely extract item from tensor or return float value."""
    if hasattr(value, 'item'):
        return value.item()
    return float(value)

class Trainer:
    """Handles the complete training process including checkpointing and validation."""
    
    def __init__(
    self,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict,
    scheduler: Optional = None,
    device: str = 'cuda',
    checkpoint_dir: str = 'checkpoints',
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    step_scheduler_batch: bool = False,
    enable_visualization: bool = True,
    visualization_dir: str = 'visualizations',
    visualization_interval: int = 5,
    enable_evaluation: bool = True
    ):
        # Initialize model and optimization components
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.current_epoch = 0
        
        # Get loss function from config
        from losses.combined import EnhancedCombinedLoss
        self.criterion = EnhancedCombinedLoss(
            ce_weight=config['losses']['ce_weight'],
            dice_weight=config['losses']['dice_weight'],
            temporal_weight=config['losses']['temporal_weight'],
            boundary_weight=config['losses'].get('boundary_weight', 0.0)
        )
        
        # Training settings - using updated PyTorch syntax
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_value = config['training'].get('grad_clip_value', 0.0)
        self.step_scheduler_batch = step_scheduler_batch
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Add result tracking
        self.validation_history = []
        self.training_history = []
        self.results_dir = Path(checkpoint_dir) / 'results'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create results files
        self.results_json = self.results_dir / 'training_results.json'
        self.results_csv = self.results_dir / 'training_metrics.csv'
        self.validation_csv = self.results_dir / 'validation_metrics.csv'
        
        # Visualization and evaluation settings
        self.enable_visualization = enable_visualization
        self.visualization_interval = visualization_interval
        self.visualization_dir = Path(visualization_dir)
        if enable_visualization:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            from utils.visualization import VideoSegmentationVisualizer
            self.visualizer = VideoSegmentationVisualizer(save_dir=self.visualization_dir)
        
        self.enable_evaluation = enable_evaluation
        if enable_evaluation:
            from utils.evaluation import DAVISEvaluator
            self.evaluator = DAVISEvaluator()
        
        self.eval_metrics = {}
        
        # Initialize CSV files with headers
        self._initialize_csv_files()



    def _initialize_csv_files(self):
        """Initialize CSV files with appropriate headers."""
        # Training metrics CSV
        if not self.results_csv.exists():
            with open(self.results_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'loss', 'ce_loss', 'dice_loss', 'boundary_loss', 'learning_rate', 'timestamp'])
        
        # Validation metrics CSV
        if not self.validation_csv.exists():
            with open(self.validation_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'val_loss', 'J_mean', 'F_mean', 'J&F', 'iou', 'f1', 'precision', 'recall', 'timestamp'])
    
    def _save_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """Save results from current epoch to files."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Save training metrics to CSV
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get('loss', 0),
                train_metrics.get('ce_loss', 0),
                train_metrics.get('dice_loss', 0),
                train_metrics.get('boundary_loss', 0),
                self.get_current_lr(),
                timestamp
            ])
        
        # Save validation metrics if available
        if val_metrics:
            with open(self.validation_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    val_metrics.get('val_loss', 0),
                    val_metrics.get('J_mean', 0),
                    val_metrics.get('F_mean', 0),
                    val_metrics.get('J&F', 0),
                    val_metrics.get('iou', 0),
                    val_metrics.get('f1', 0),
                    val_metrics.get('precision', 0),
                    val_metrics.get('recall', 0),
                    timestamp
                ])
        
        # Update history
        train_record = {'epoch': epoch, 'timestamp': timestamp, **train_metrics}
        self.training_history.append(train_record)
        
        if val_metrics:
            val_record = {'epoch': epoch, 'timestamp': timestamp, **val_metrics}
            self.validation_history.append(val_record)
        
        # Save complete history to JSON
        complete_results = {
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'best_val_loss': float(self.best_val_loss),
            'current_epoch': epoch,
            'last_updated': timestamp
        }
        
        with open(self.results_json, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def _create_training_summary(self):
        """Create a comprehensive training summary."""
        summary_file = self.results_dir / 'training_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Basic info
            f.write(f"Total epochs trained: {len(self.training_history)}\n")
            f.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
            f.write(f"Final learning rate: {self.get_current_lr():.2e}\n\n")
            
            # Training metrics progression
            if self.training_history:
                f.write("TRAINING METRICS PROGRESSION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Epoch':<8} {'Loss':<10} {'Dice':<10} {'LR':<12}\n")
                f.write("-" * 40 + "\n")
                
                # Show every 10th epoch + first/last few
                epochs_to_show = []
                if len(self.training_history) <= 20:
                    epochs_to_show = list(range(len(self.training_history)))
                else:
                    # First 5, every 10th, last 5
                    epochs_to_show.extend(range(5))
                    epochs_to_show.extend(range(10, len(self.training_history)-5, 10))
                    epochs_to_show.extend(range(len(self.training_history)-5, len(self.training_history)))
                
                for i in epochs_to_show:
                    record = self.training_history[i]
                    f.write(f"{record['epoch']:<8} {record.get('loss', 0):<10.4f} {record.get('dice_loss', 0):<10.4f} {record.get('learning_rate', 0):<12.2e}\n")
            
            # Validation metrics progression
            if self.validation_history:
                f.write("\n\nVALIDATION METRICS PROGRESSION:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Epoch':<8} {'Val Loss':<10} {'J&F':<8} {'IoU':<8} {'F1':<8}\n")
                f.write("-" * 60 + "\n")
                
                for record in self.validation_history[-10:]:  # Last 10 validation results
                    f.write(f"{record['epoch']:<8} {record.get('val_loss', 0):<10.4f} {record.get('J&F', 0):<8.4f} {record.get('iou', 0):<8.4f} {record.get('f1', 0):<8.4f}\n")
            
            # Best results
            if self.validation_history:
                best_metrics = max(self.validation_history, key=lambda x: x.get('J&F', 0))
                f.write(f"\n\nBEST VALIDATION RESULTS (Epoch {best_metrics['epoch']}):\n")
                f.write("-" * 40 + "\n")
                for key, value in best_metrics.items():
                    if key not in ['epoch', 'timestamp'] and isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.6f}\n")
        
        self.logger.info(f"Training summary saved to {summary_file}")
    
    def get_current_lr(self):
        """Get the current learning rate from the optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def save_checkpoint(self, metrics: Dict[str, float], name: str = 'model') -> None:
        """Saves a checkpoint of the current training state."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        save_path = self.checkpoint_dir / f'{name}.pth'
        torch.save(checkpoint, save_path)
        
        # Also save metrics separately for easy access
        import json
        metrics_path = self.checkpoint_dir / f'{name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        
        self.logger.info(f"Saved checkpoint and metrics to {self.checkpoint_dir}")
    
    def load_checkpoint(self, path: str, load_best: bool = True) -> None:
        """Loads a checkpoint and restores the training state."""
        path = Path(path)
        if load_best and not path.name.startswith('model_best'):
            path = path.parent / f'model_best.pth'
        
        if not path.exists():
            self.logger.warning(f"No checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        # Restore scheduler if it exists
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Restored checkpoint from {path} (epoch {self.epoch})")
    
    def _reset_temporal_states(self, module):
        """Reset temporal states in stateful modules."""
        if hasattr(module, 'features') and isinstance(module.features, deque):
            module.features.clear()
    
    def train_epoch(self, train_loader):
        """Run a single training epoch with memory and performance optimizations."""
        self.model.train()
        
        # Initialize tracking variables
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        running_boundary_loss = 0.0
        running_samples = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=len(train_loader), desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
                masks = batch['masks'].to(self.device)    # [B, T, H, W]
                
                # Increment sample count
                batch_size = frames.shape[0]
                running_samples += batch_size
                
                # Free memory explicitly
                torch.cuda.empty_cache()
                
                # Forward pass with mixed precision - updated to new PyTorch syntax
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(frames)
                        loss_dict = self.criterion(outputs, {'masks': masks})
                        
                        # Extract loss components
                        loss = loss_dict['loss']
                        ce_loss = loss_dict.get('ce_loss', 0.0)
                        dice_loss = loss_dict.get('dice_loss', 0.0)
                        boundary_loss = loss_dict.get('boundary_loss', 0.0)
                else:
                    # Standard forward pass without mixed precision
                    outputs = self.model(frames)
                    loss_dict = self.criterion(outputs, {'masks': masks})
                    
                    # Extract loss components
                    loss = loss_dict['loss']
                    ce_loss = loss_dict.get('ce_loss', 0.0)
                    dice_loss = loss_dict.get('dice_loss', 0.0)
                    boundary_loss = loss_dict.get('boundary_loss', 0.0)
                
                # 🚀 ROBUST ADAPTIVE THRESHOLDING MONITORING
                if batch_idx % 50 == 0:  # Log every 50 batches
                    try:
                        # Check if model has adaptive masks
                        if 'adaptive_masks' in outputs:
                            pred_positive_ratio = outputs['adaptive_masks'].float().mean().item()
                            raw_positive_ratio = (outputs['pred_masks'] > 0.5).float().mean().item()
                            
                            self.logger.info(f"Batch {batch_idx}: Raw fg ratio: {raw_positive_ratio:.4f}, "
                                        f"Adaptive fg ratio: {pred_positive_ratio:.4f}")
                        else:
                            pred_positive_ratio = (outputs['pred_masks'] > 0.5).float().mean().item()
                            raw_positive_ratio = pred_positive_ratio
                            
                            self.logger.info(f"Batch {batch_idx}: Pred fg ratio: {pred_positive_ratio:.4f}")
                        
                        gt_positive_ratio = (masks > 0).float().mean().item()
                        self.logger.info(f"Batch {batch_idx}: GT fg ratio: {gt_positive_ratio:.4f}")
                        
                        # Safe ratio checking
                        min_threshold = 0.001  # Minimum meaningful foreground ratio
                        
                        if gt_positive_ratio > min_threshold and pred_positive_ratio > min_threshold:
                            # Both have meaningful foreground - compare ratios
                            ratio = pred_positive_ratio / gt_positive_ratio
                            if ratio > 3.0:
                                self.logger.warning(f"Overpredicting foreground! Ratio: {ratio:.2f}x")
                            elif ratio < 0.33:
                                self.logger.warning(f"Underpredicting foreground! Ratio: {ratio:.2f}x")
                            else:
                                self.logger.info(f"Foreground ratio balanced! Pred/GT ratio: {ratio:.2f}x")
                        elif gt_positive_ratio <= min_threshold and pred_positive_ratio > 0.05:
                            # GT has no foreground but model predicts a lot
                            self.logger.warning(f"GT has no foreground, but model predicts {pred_positive_ratio:.3f}")
                        elif gt_positive_ratio > min_threshold and pred_positive_ratio <= min_threshold:
                            # GT has foreground but model predicts none
                            self.logger.warning(f"GT has {gt_positive_ratio:.3f} foreground, but model predicts none")
                        else:
                            # Both have minimal foreground
                            self.logger.info(f"Both GT and prediction have minimal foreground")
                            
                    except Exception as e:
                        self.logger.error(f"Error in monitoring: {e}")
                        # Continue training even if monitoring fails
                        pass
                
                # 🚀 ADD GRADIENT MONITORING (every 10 batches)
                if batch_idx % 10 == 0:
                    self._log_gradients()
                
                # Handle gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    # Scale loss
                    scaled_loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.mixed_precision:
                        self.scaler.scale(scaled_loss).backward()
                        
                        # Update weights after accumulating enough gradients
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            # Apply gradient clipping if configured
                            if self.grad_clip_value > 0:
                                self.scaler.unscale_(self.optimizer)
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                            
                            # Step optimizer and scaler
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                    else:
                        scaled_loss.backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            if self.grad_clip_value > 0:
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                            
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                    
                    # Update scheduler if batch-based
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.scheduler is not None and self.step_scheduler_batch:
                            self.scheduler.step()
                else:
                    # Standard backward and update without accumulation
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                        
                        # Apply gradient clipping if configured
                        if self.grad_clip_value > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                        
                        # Step optimizer and scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        loss.backward()
                        
                        if self.grad_clip_value > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    # Update scheduler if batch-based
                    if self.scheduler is not None and self.step_scheduler_batch:
                        self.scheduler.step()
                
                # Update loss tracking with proper detachment and moving to CPU
                running_loss += get_item_safely(loss) * batch_size
                running_ce_loss += get_item_safely(ce_loss) * batch_size
                running_dice_loss += get_item_safely(dice_loss) * batch_size
                running_boundary_loss += get_item_safely(boundary_loss) * batch_size
                
                # 🚀 ENHANCED PROGRESS BAR with more info
                postfix_dict = {
                    'loss': f"{get_item_safely(loss):.4f}",
                    'dice': f"{get_item_safely(dice_loss):.4f}",
                    'lr': f"{self.get_current_lr():.6f}"
                }
                
                # Add adaptive ratio to progress bar if available
                if 'adaptive_masks' in outputs:
                    adaptive_ratio = outputs['adaptive_masks'].float().mean().item()
                    postfix_dict['fg_ratio'] = f"{adaptive_ratio:.3f}"
                
                pbar.set_postfix(postfix_dict)
                pbar.update(1)
                
                # Clean up memory
                del frames, masks, outputs, loss_dict
                
                # Update global step counter
                self.global_step += 1
        
        # Calculate average metrics
        avg_loss = running_loss / running_samples
        avg_ce_loss = running_ce_loss / running_samples
        avg_dice_loss = running_dice_loss / running_samples
        avg_boundary_loss = running_boundary_loss / running_samples
        
        # Update epoch-based scheduler
        if self.scheduler is not None and not self.step_scheduler_batch:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        # Return metrics
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'dice_loss': avg_dice_loss,
            'boundary_loss': avg_boundary_loss
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model with memory-efficient processing."""
        self.model.eval()
        total_loss = 0.0
        
        # Accumulate predictions and ground truth for metrics
        all_predictions = []
        all_ground_truths = []
        all_sequences = []
        
        # Process in smaller batches if needed
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            try:
                # Get data
                frames = batch['frames'].to(self.device)
                masks = batch['masks'].to(self.device)
                sequence = batch.get('sequence', [f"seq_{batch_idx}"])
                
                # Forward pass
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(frames)
                        loss_dict = self.criterion(outputs, {'masks': masks})
                        loss = loss_dict['loss']
                else:
                    outputs = self.model(frames)
                    loss_dict = self.criterion(outputs, {'masks': masks})
                    loss = loss_dict['loss']
                
                # 🚀 ADAPTIVE THRESHOLDING DEBUG
                if batch_idx == 0:  # Log first batch of validation
                    self.logger.info("="*60)
                    self.logger.info("VALIDATION DEBUG - ADAPTIVE THRESHOLDING")
                    self.logger.info("="*60)
                    
                    if 'adaptive_masks' in outputs:
                        raw_fg = (outputs['pred_masks'] > 0.5).float().mean().item()
                        adaptive_fg = outputs['adaptive_masks'].float().mean().item()
                        gt_fg = (masks > 0).float().mean().item()
                        
                        self.logger.info(f"🎯 OVERALL BATCH STATS:")
                        self.logger.info(f"   Raw FG ratio: {raw_fg:.4f}")
                        self.logger.info(f"   Adaptive FG ratio: {adaptive_fg:.4f}")
                        self.logger.info(f"   GT FG ratio: {gt_fg:.4f}")
                        
                        # Calculate improvement
                        if gt_fg > 0.001:
                            raw_error = abs(raw_fg - gt_fg) / gt_fg
                            adaptive_error = abs(adaptive_fg - gt_fg) / gt_fg
                            self.logger.info(f"   Raw error: {raw_error:.2f}x GT")
                            self.logger.info(f"   Adaptive error: {adaptive_error:.2f}x GT")
                            
                            if adaptive_error < raw_error:
                                self.logger.info("   ✅ Adaptive thresholding is helping!")
                            else:
                                self.logger.info("   ⚠️  Adaptive thresholding not improving")
                        
                        # Check individual frames
                        self.logger.info(f"📋 FRAME-BY-FRAME BREAKDOWN:")
                        num_frames = min(3, outputs['adaptive_masks'].shape[1])  # Check first 3 frames
                        for t in range(num_frames):
                            frame_raw_fg = (outputs['pred_masks'][0, t] > 0.5).float().mean().item()
                            frame_adaptive_fg = outputs['adaptive_masks'][0, t].float().mean().item()
                            frame_gt_fg = (masks[0, t] > 0).float().mean().item()
                            
                            self.logger.info(f"   Frame {t}: Raw={frame_raw_fg:.4f}, "
                                        f"Adaptive={frame_adaptive_fg:.4f}, GT={frame_gt_fg:.4f}")
                            
                            # Check if adaptive is closer to GT
                            if frame_gt_fg > 0.001:
                                raw_diff = abs(frame_raw_fg - frame_gt_fg)
                                adaptive_diff = abs(frame_adaptive_fg - frame_gt_fg)
                                if adaptive_diff < raw_diff:
                                    self.logger.info(f"             ✅ Frame {t}: Adaptive closer to GT")
                                else:
                                    self.logger.info(f"             ❌ Frame {t}: Raw closer to GT")
                        
                        # Check shapes match
                        self.logger.info(f"📐 SHAPE CHECK:")
                        self.logger.info(f"   Pred masks shape: {outputs['pred_masks'].shape}")
                        self.logger.info(f"   Adaptive masks shape: {outputs['adaptive_masks'].shape}")
                        self.logger.info(f"   GT masks shape: {masks.shape}")
                        
                    else:
                        self.logger.error("❌ CRITICAL: No adaptive_masks found in outputs!")
                        self.logger.error(f"   Available output keys: {list(outputs.keys())}")
                        self.logger.error("   Adaptive thresholding is NOT working!")
                        
                        # Fallback info
                        raw_fg = (outputs['pred_masks'] > 0.5).float().mean().item()
                        gt_fg = (masks > 0).float().mean().item()
                        self.logger.info(f"   Raw FG ratio: {raw_fg:.4f}")
                        self.logger.info(f"   GT FG ratio: {gt_fg:.4f}")
                    
                    self.logger.info("="*60)
                
                # Track loss
                total_loss += loss.item() * frames.shape[0]
                
                # Store predictions for evaluation (detach and move to CPU to save memory)
                if self.enable_evaluation:
                    # Use adaptive masks if available, otherwise fall back to regular pred_masks
                    masks_to_use = outputs.get('adaptive_masks', outputs['pred_masks'])
                    all_predictions.append(masks_to_use[0].cpu())
                    all_ground_truths.append(masks[0].cpu())
                    all_sequences.append(sequence[0] if isinstance(sequence, list) else sequence)
                
                # Visualize predictions occasionally
                if self.enable_visualization and batch_idx % self.visualization_interval == 0:
                    # Use adaptive masks for visualization if available
                    vis_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                    self._visualize_prediction(
                        frames=frames[0].cpu(),
                        pred_masks=vis_masks[0].cpu(),
                        gt_masks=masks[0].cpu(),
                        sequence_name=f"{sequence[0] if isinstance(sequence, list) else sequence}_epoch_{self.current_epoch}"
                    )
                
                # Free memory
                del frames, masks, outputs, loss_dict
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"Out of memory during validation. Skipping batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader.dataset)
        
        # Calculate evaluation metrics if enabled
        metrics = {'val_loss': avg_loss}
        
        if self.enable_evaluation and all_predictions:
            # Evaluate models using DAVIS metrics
            try:
                eval_results = self.evaluator.evaluate_binary_segmentation(
                    predictions=all_predictions,
                    ground_truths=all_ground_truths,
                    sequence_names=all_sequences
                )
                
                # Add global metrics to result
                for key, value in eval_results['global'].items():
                    metrics[key] = value
                
                # Save evaluation metrics for future reference
                self.eval_metrics = metrics
                
            except Exception as e:
                self.logger.error(f"Error during evaluation: {str(e)}")
        
        return metrics
    
    def _visualize_prediction(self, frames, pred_masks, gt_masks, sequence_name):
        """Create and save visualization for a prediction."""
        if not self.enable_visualization:
            return
            
        try:
            self.visualizer.visualize_sequence(
                frames=frames,
                pred_masks=pred_masks,
                gt_masks=gt_masks,
                sequence_name=sequence_name,
                max_frames=min(4, frames.shape[0])  # Limit number of frames for efficiency
            )
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
    
    def train(self, train_loader, val_loader=None, num_epochs=100, validate_every=1, save_every=10, patience=15):
        """Main training loop with comprehensive result saving."""
        self.logger.info(f"Starting training from epoch {self.epoch}")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Early stopping variables
        best_val_loss = float('inf')
        no_improvement_count = 0
        
        for epoch in range(self.epoch, num_epochs):
            # Reset temporal state at the start of each epoch
            self.model.apply(self._reset_temporal_states)
            
            # Update epoch counters
            self.epoch = epoch
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch} training - Loss: {train_metrics['loss']:.4f}")
            
            # Validation phase
            val_metrics = None
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                
                # Log validation results
                self.logger.info(f"Epoch {epoch} validation - Loss: {val_loss:.4f}")
                if 'J&F' in val_metrics:
                    self.logger.info(f"  J&F: {val_metrics['J&F']:.4f}, IoU: {val_metrics.get('iou', 0):.4f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    
                    # Save best model
                    self.logger.info(f"Validation loss improved by {improvement:.6f}. Saving model...")
                    self.save_checkpoint(val_metrics, name='model_best')
                else:
                    no_improvement_count += 1
                    self.logger.info(f"No improvement for {no_improvement_count} epochs.")
                    
                    # Early stopping check
                    if no_improvement_count >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
                
                # Regular checkpoint saving
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(val_metrics, name=f'model_epoch_{epoch}')
            
            # Save epoch results to files
            self._save_epoch_results(epoch, train_metrics, val_metrics)
        
        # Create final training summary
        self._create_training_summary()
        
        self.logger.info("Training completed!")
        self.logger.info(f"All results saved to: {self.results_dir}")
        
        # Load best model at the end
        self.load_checkpoint(os.path.join(self.checkpoint_dir, 'model_best.pth'))
        
        return best_val_loss
    
    def find_learning_rate(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1,
        num_iterations: int = 100,
        step_mode: str = "exp"
    ):
        """Find optimal learning rate using the learning rate range test."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Save original model and optimizer state
        old_state_dict = copy.deepcopy(self.model.state_dict())
        old_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        
        # Initialize learning rate and lists to track values
        if step_mode == "exp":
            lr_factor = (end_lr / start_lr) ** (1 / (num_iterations - 1))
            lr_schedule = [start_lr * (lr_factor ** i) for i in range(num_iterations)]
        else:  # Linear schedule
            lr_schedule = np.linspace(start_lr, end_lr, num_iterations)
        
        losses = []
        log_lrs = []
        best_loss = float('inf')
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        
        # Interactive plot setup
        plt.figure(figsize=(10, 6))
        plt.ion()
        ax = plt.gca()
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_xscale('log')
        line, = ax.plot([], [], 'b-')
        
        # Run learning rate finder
        iterator = iter(train_loader)
        for i, lr in enumerate(tqdm(lr_schedule, desc="Finding optimal learning rate")):
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            # Move data to device
            frames = batch['frames'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            # Forward pass - using old PyTorch syntax
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(frames)
                    loss_dict = self.criterion(outputs, {'masks': masks})
                    loss = loss_dict['loss']
            else:
                outputs = self.model(frames)
                loss_dict = self.criterion(outputs, {'masks': masks})
                loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            if batch_idx % 10 == 0:  # Log every 10 batches
                self._log_gradients()
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Record values for plotting
            losses.append(loss.item())
            log_lrs.append(math.log10(lr))
            
            # Update interactive plot
            line.set_data(log_lrs, losses)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            
            # Check for divergence
            if i > 0 and loss.item() > 4 * best_loss:
                break
            
            if loss.item() < best_loss:
                best_loss = loss.item()
        
        # Restore original model and optimizer state
        self.model.load_state_dict(old_state_dict)
        self.optimizer.load_state_dict(old_optimizer_state_dict)
        
        # Find suggested learning rate (point of steepest descent)
        # This is the point where the loss is decreasing the fastest
        derivatives = [(losses[i+1] - losses[i]) / (log_lrs[i+1] - log_lrs[i]) 
                      for i in range(len(losses)-1)]
        min_derivative_idx = np.argmin(derivatives)
        suggested_lr = 10 ** log_lrs[min_derivative_idx]
        
        # Finalize plot
        plt.ioff()
        plt.figure(figsize=(10, 6))
        plt.plot(log_lrs, losses)
        plt.scatter([log_lrs[min_derivative_idx]], [losses[min_derivative_idx]], 
                   color='red', s=100, marker='o')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.xscale('log')
        plt.axvline(x=log_lrs[min_derivative_idx], color='r', linestyle='--')
        plt.title(f'Learning Rate Finder - Suggested LR: {suggested_lr:.1e}')
        plt.savefig(os.path.join(self.checkpoint_dir, 'lr_finder.png'))
        plt.close()
        
        self.logger.info(f"Suggested learning rate: {suggested_lr:.1e}")
        
        return suggested_lr

    def evaluate(self, val_loader, visualize=True):
        """Evaluate model on validation data."""
        self.model.eval()
        
        # Initialize tracking variables
        total_loss = 0.0
        all_predictions = []
        all_ground_truths = []
        all_sequences = []
        
        # Evaluate without gradient computation
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
                # Get data
                frames = batch['frames'].to(self.device)
                masks = batch['masks'].to(self.device)
                sequence = batch.get('sequence', [f"seq_{batch_idx}"])
                
                # Forward pass
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(frames)
                        loss_dict = self.criterion(outputs, {'masks': masks})
                        loss = loss_dict['loss']
                else:
                    outputs = self.model(frames)
                    loss_dict = self.criterion(outputs, {'masks': masks})
                    loss = loss_dict['loss']
                
                # Track loss
                total_loss += loss.item() * frames.shape[0]
                
                # 🚀 CHANGE THIS: Store predictions using adaptive masks
                masks_to_use = outputs.get('adaptive_masks', outputs['pred_masks'])
                all_predictions.append(masks_to_use[0].cpu())
                all_ground_truths.append(masks[0].cpu())
                all_sequences.append(sequence[0] if isinstance(sequence, list) else sequence)
                
                # Create visualizations for specific batches
                if visualize and batch_idx % self.visualization_interval == 0:
                    # 🚀 ALSO UPDATE VISUALIZATION
                    vis_masks = outputs.get('adaptive_masks', outputs['pred_masks'])
                    self._visualize_prediction(
                        frames=frames[0].cpu(),
                        pred_masks=vis_masks[0].cpu(),
                        gt_masks=masks[0].cpu(),
                        sequence_name=f"{sequence[0]}_final_eval"
                    )
                
                # Clean up memory
                del frames, masks, outputs
                torch.cuda.empty_cache()
    
    
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader.dataset)
        
        # Calculate metrics
        metrics = {'loss': avg_loss}
        
        if self.enable_evaluation and all_predictions:
            # Get comprehensive evaluation metrics
            eval_results = self.evaluator.evaluate_binary_segmentation(
                predictions=all_predictions,
                ground_truths=all_ground_truths,
                sequence_names=all_sequences
            )
            
            # Add global metrics to results
            for key, value in eval_results['global'].items():
                metrics[key] = value
            
            # Print evaluation summary
            self.logger.info("\nEvaluation Results:")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.4f}")
        
        return metrics
    

    def _log_gradients(self):
        """Log gradient statistics for each parameter."""
        total_norm = 0.0
        param_norms = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.norm(2).item()
                param_norms.append((name, norm))
                total_norm += norm ** 2
        total_norm = total_norm ** 0.5
        
        # Log the highest gradient norms
        param_norms.sort(key=lambda x: x[1], reverse=True)
        self.logger.info(f"Total gradient norm: {total_norm:.4f}")
        for name, norm in param_norms[:5]:  # Log top 5
            self.logger.info(f"Gradient norm for {name}: {norm:.4f}")
        
        # Also check for any dead gradients (exact zeros)
        zero_grads = [(name, norm) for name, norm in param_norms if norm == 0]
        if zero_grads:
            self.logger.warning(f"Found {len(zero_grads)} parameters with zero gradients")
            for name, _ in zero_grads[:3]:  # Log first few
                self.logger.warning(f"Zero gradient for: {name}")

    def test_model_speed(self, frames, num_iterations=10):
        """Test the model's forward and backward pass speed."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # Create dummy target
        B, T, C, H, W = frames.shape
        target = torch.rand(B, T, 1, H, W).to(frames.device)
        target = (target > 0.5).float()
        
        # Warmup
        for _ in range(2):
            outputs = self.model(frames)
            loss = criterion(outputs['logits'], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            outputs = self.model(frames)
            loss = criterion(outputs['logits'], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = (B * T) / avg_time
        
        print(f"Average iteration time: {avg_time:.4f} seconds")
        print(f"Frames per second: {fps:.2f}")
        print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        return avg_time, fps