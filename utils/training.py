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
        self.config = config  # Store config for LR access
        
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

    def get_current_lr(self):
        """Get the current learning rate from the optimizer."""
        try:
            for param_group in self.optimizer.param_groups:
                return param_group['lr']
        except Exception as e:
            print(f"Error getting LR: {e}")
            return 0.0

    def _get_item_safely(self, value):
        """Safely extract item from tensor or return float value."""
        if hasattr(value, 'item'):
            return value.item()
        return float(value)

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
    
    # In utils/training.py, replace the _save_epoch_results method with this robust version:

    def _save_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """Save results from current epoch to files with error handling."""
        timestamp = datetime.datetime.now().isoformat()
        
        try:
            # Save training metrics to CSV
            with open(self.results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    self._safe_float(train_metrics.get('loss', 0)),
                    self._safe_float(train_metrics.get('ce_loss', 0)),
                    self._safe_float(train_metrics.get('dice_loss', 0)),
                    self._safe_float(train_metrics.get('boundary_loss', 0)),
                    self._safe_float(self.get_current_lr()),
                    timestamp
                ])
        except Exception as e:
            self.logger.error(f"Failed to save training metrics CSV: {e}")
        
        try:
            # Save validation metrics if available
            if val_metrics:
                with open(self.validation_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        self._safe_float(val_metrics.get('val_loss', 0)),
                        self._safe_float(val_metrics.get('J_mean', 0)),
                        self._safe_float(val_metrics.get('F_mean', 0)),
                        self._safe_float(val_metrics.get('J&F', 0)),
                        self._safe_float(val_metrics.get('iou', 0)),
                        self._safe_float(val_metrics.get('f1', 0)),
                        self._safe_float(val_metrics.get('precision', 0)),
                        self._safe_float(val_metrics.get('recall', 0)),
                        timestamp
                    ])
        except Exception as e:
            self.logger.error(f"Failed to save validation metrics CSV: {e}")
        
        try:
            # Update history with safe conversion
            train_record = {
                'epoch': epoch, 
                'timestamp': timestamp,
                **{k: self._safe_float(v) for k, v in train_metrics.items()}
            }
            self.training_history.append(train_record)
            
            if val_metrics:
                val_record = {
                    'epoch': epoch, 
                    'timestamp': timestamp,
                    **{k: self._safe_float(v) for k, v in val_metrics.items()}
                }
                self.validation_history.append(val_record)
        except Exception as e:
            self.logger.error(f"Failed to update history: {e}")
        
        try:
            # Save complete history to JSON with safe conversion
            complete_results = {
                'training_history': self.training_history,
                'validation_history': self.validation_history,
                'best_val_loss': self._safe_float(self.best_val_loss),
                'current_epoch': epoch,
                'last_updated': timestamp
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.results_json.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(complete_results, f, indent=2, default=self._json_serializer)
            
            # Atomic rename
            temp_file.rename(self.results_json)
            
            self.logger.info(f"Results saved to {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            # Try to save a minimal version
            try:
                minimal_results = {
                    'current_epoch': epoch,
                    'last_updated': timestamp,
                    'error': f"Full save failed: {str(e)}"
                }
                with open(self.results_json, 'w') as f:
                    json.dump(minimal_results, f, indent=2)
            except Exception as e2:
                self.logger.error(f"Failed to save minimal JSON: {e2}")

    def _safe_float(self, value):
        """Safely convert value to float, handling tensors and edge cases."""
        try:
            if hasattr(value, 'item'):
                val = value.item()
            else:
                val = float(value)
            
            # Handle NaN and infinity
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return val
        except (ValueError, TypeError):
            return 0.0

    def _json_serializer(self, obj):
        """Custom JSON serializer for handling special types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            val = float(obj)
            return 0.0 if (np.isnan(val) or np.isinf(val)) else val
        elif hasattr(obj, 'item'):  # Torch tensor
            return obj.item()
        else:
            return str(obj)  # Fallback to string representation

    def train_epoch(self, train_loader):
        """Run a single training epoch with emergency LR fix and memory optimizations."""
        self.model.train()
        
        # 🚨 EMERGENCY LR CHECK AND FIX - ALWAYS AT START OF EPOCH
        current_lr = self.get_current_lr()
        if current_lr == 0 or current_lr < 1e-8:
            print(f"\n🚨 EMERGENCY: Learning rate is {current_lr:.2e}")
            print("Applying emergency fix...")
            
            # Get target LR from config or use sensible default
            if hasattr(self, 'config') and 'optimizer' in self.config:
                target_lr = self.config['optimizer'].get('lr', 5e-5)
            else:
                target_lr = 5e-5  # Safe default
            
            # Ensure target LR is reasonable
            if target_lr == 0 or target_lr < 1e-8:
                target_lr = 5e-5
                print(f"Config LR was also zero, using fallback: {target_lr:.2e}")
            
            # Fix the learning rate
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = param_group['lr']
                param_group['lr'] = target_lr
                print(f"✅ Fixed param group {i} LR: {old_lr:.2e} → {target_lr:.2e}")
            
            # Reset scheduler if it exists and is problematic
            if self.scheduler is not None:
                print("🔄 Resetting scheduler...")
                try:
                    if hasattr(self.scheduler, '_step_count'):
                        self.scheduler._step_count = 0
                    if hasattr(self.scheduler, 'last_epoch'):
                        self.scheduler.last_epoch = -1
                    print("✅ Scheduler reset successful")
                except Exception as e:
                    print(f"⚠️ Scheduler reset failed: {e}")
            
            print(f"✅ Emergency fix complete. New LR: {self.get_current_lr():.2e}")
        else:
            print(f"✅ LR check passed: {current_lr:.2e}")
        
        # Initialize tracking variables
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        running_boundary_loss = 0.0
        running_samples = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=len(train_loader), desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # 🚨 CONTINUOUS LR MONITORING - Check every 50 batches
                if batch_idx % 50 == 0:
                    current_lr = self.get_current_lr()
                    if current_lr == 0 or current_lr < 1e-8:
                        print(f"\n🚨 Mid-training LR emergency at batch {batch_idx}")
                        target_lr = 5e-5
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = target_lr
                        print(f"⚡ Fixed LR: {current_lr:.2e} → {target_lr:.2e}")
                
                # Move data to device
                frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
                masks = batch['masks'].to(self.device)    # [B, T, H, W]
                
                # Increment sample count
                batch_size = frames.shape[0]
                running_samples += batch_size
                
                # Free memory explicitly
                torch.cuda.empty_cache()
                
                # Forward pass with mixed precision - updated to new PyTorch syntax
                try:
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
                    
                    # Check for NaN/Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ WARNING: Invalid loss detected at batch {batch_idx}")
                        print(f"Loss: {loss}, CE: {ce_loss}, Dice: {dice_loss}")
                        continue  # Skip this batch
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"🔥 OOM at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
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
                            try:
                                self.scheduler.step()
                            except Exception as e:
                                print(f"⚠️ Scheduler step failed: {e}")
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
                        try:
                            self.scheduler.step()
                        except Exception as e:
                            print(f"⚠️ Scheduler step failed: {e}")
                
                # Update loss tracking with proper detachment and moving to CPU
                running_loss += self._get_item_safely(loss) * batch_size
                running_ce_loss += self._get_item_safely(ce_loss) * batch_size
                running_dice_loss += self._get_item_safely(dice_loss) * batch_size
                running_boundary_loss += self._get_item_safely(boundary_loss) * batch_size
                
                # 🚀 ENHANCED PROGRESS BAR with LR monitoring
                current_lr = self.get_current_lr()
                postfix_dict = {
                    'loss': f"{self._get_item_safely(loss):.4f}",
                    'dice': f"{self._get_item_safely(dice_loss):.4f}",
                    'lr': f"{current_lr:.6f}"
                }
                
                # 🚨 EMERGENCY LR FIX DURING TRAINING
                if current_lr == 0 or current_lr < 1e-8:
                    # Emergency fix during training
                    target_lr = 5e-5  # Adjust as needed
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = target_lr
                    print(f"\n⚡ Mid-training LR fix: {current_lr:.2e} → {target_lr:.2e}")
                    postfix_dict['lr'] = f"{target_lr:.6f}"
                
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
            try:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            except Exception as e:
                print(f"⚠️ Epoch scheduler step failed: {e}")
        
        # Final LR check
        final_lr = self.get_current_lr()
        print(f"Epoch {self.current_epoch} completed. Final LR: {final_lr:.2e}")
        
        # Return metrics
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'dice_loss': avg_dice_loss,
            'boundary_loss': avg_boundary_loss,
            'learning_rate': final_lr  # Add LR to returned metrics
        }

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
        
        self.logger.info("Training completed!")
        self.logger.info(f"All results saved to: {self.results_dir}")
        
        # Load best model at the end
        self.load_checkpoint(os.path.join(self.checkpoint_dir, 'model_best.pth'))
        
        return best_val_loss

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
                
                # Store predictions using adaptive masks
                masks_to_use = outputs.get('adaptive_masks', outputs['pred_masks'])
                all_predictions.append(masks_to_use[0].cpu())
                all_ground_truths.append(masks[0].cpu())
                all_sequences.append(sequence[0] if isinstance(sequence, list) else sequence)
                
                # Create visualizations for specific batches
                if visualize and batch_idx % self.visualization_interval == 0:
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