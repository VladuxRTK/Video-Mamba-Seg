

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
from tqdm import tqdm

from losses.combined import CombinedLoss
from losses.temporal_consistency import TemporalConsistencyLoss

from losses.segmentation import BinarySegmentationLoss

import torch
import torch.nn as nn
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Union
from models.backbone import TemporalFeatureBank


# At the top of your file, outside any class
def get_item_safely(value):
    """Safely extract item from tensor or return float value."""
    if hasattr(value, 'item'):
        return value.item()
    return value

class Trainer:
    """Handles the complete training process including checkpointing and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        step_scheduler_batch: bool = False,  # Add this parameter
        # Add new parameters for visualization and evaluation
        enable_visualization: bool = True,
        visualization_dir: str = 'visualizations',
        visualization_interval: int = 5,
        enable_evaluation: bool = True
    ):
        # Existing initialization code
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
        
        # Replace the current criterion with binary segmentation loss
        # Initialize the loss function with the weights provided in the constructor
        self.criterion = BinarySegmentationLoss(
            ce_weight=config['losses']['ce_weight'],
            dice_weight=config['losses']['dice_weight'],
            boundary_weight=config['losses'].get('boundary_weight', 0.0)  # Add default for backward compatibility
        )
        # Modern mixed precision setup
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_metrics = {}

        self.grad_clip_value = config['training'].get('grad_clip_value', 0.0)
        self.step_scheduler_batch = step_scheduler_batch  # Use the parameter value
    
    # Rest of your initialization code...
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize visualization and evaluation tools if enabled
        self.enable_visualization = enable_visualization
        self.visualization_interval = visualization_interval
        if enable_visualization:
            from utils.visualization import VideoSegmentationVisualizer
            self.visualization_dir = Path(visualization_dir)
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            self.visualizer = VideoSegmentationVisualizer(save_dir=self.visualization_dir)
        
        self.enable_evaluation = enable_evaluation
        if enable_evaluation:
            from utils.evaluation import DAVISEvaluator
            self.evaluator = DAVISEvaluator()

    
    def get_current_lr(self):
        """Get the current learning rate from the optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def current_epoch(self):
        """Get the current training epoch."""
        return getattr(self, '_current_epoch', 0)
        
    @current_epoch.setter
    def current_epoch(self, epoch):
        """Set the current training epoch."""
        self._current_epoch = epoch
    
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
        metrics_path = self.checkpoint_dir / f'{name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved checkpoint and metrics to {self.checkpoint_dir}")
    
    def load_checkpoint(self, path: str, load_best: bool = True) -> None:
        """Loads a checkpoint and restores the training state."""
        path = Path(path)
        if load_best:
            path = path.parent / f'{path.stem}_best.pth'
        
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
    
    
    def train_epoch(self, train_loader):
        """
        Run a single training epoch with binary segmentation handling.
        
        This method processes each batch of video data, computes the loss,
        performs backpropagation, and tracks training metrics throughout the epoch.
        
        Args:
            train_loader: DataLoader providing training batches
            
        Returns:
            Dictionary containing average loss values and metrics for the epoch
        """
        self.model.train()
        
        # Initialize tracking variables
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        total_boundary_loss = 0.0  # Track boundary loss
        
        # Progress metrics
        batch_count = len(train_loader)
        processed_samples = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=batch_count, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
                masks = batch['masks'].to(self.device)    # [B, T, H, W]
                
                # Track batch size for averaging
                batch_size = frames.shape[0]
                processed_samples += batch_size
                
                # Mixed precision training if enabled
                with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                    # Forward pass - model outputs dict with 'logits' and 'pred_masks'
                    outputs = self.model(frames)
                    
                    # Compute loss - expects dict with 'masks' key
                    loss_dict = self.criterion(outputs, {'masks': masks})
                    
                    # Get individual loss components
                    loss = loss_dict['loss']  # Total loss
                    ce_loss = loss_dict.get('ce_loss', 0.0)
                    dice_loss = loss_dict.get('dice_loss', 0.0)
                    boundary_loss = loss_dict.get('boundary_loss', 0.0)  # Get boundary loss
                
                # Backward pass with gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    # Only update weights after accumulating enough gradients
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Unscale gradients for clipping (if used)
                        if self.grad_clip_value > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                        
                        # Optimizer step with scaler for mixed precision
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                        
                        # Step the scheduler if it's batch-based
                        if self.scheduler is not None and self.step_scheduler_batch:
                            self.scheduler.step()
                else:
                    # Standard backward and update (no accumulation)
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping if configured
                    if self.grad_clip_value > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Step the scheduler if it's batch-based
                    if self.scheduler is not None and self.step_scheduler_batch:
                        self.scheduler.step()
                
                # Update tracking metrics
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss * batch_size if isinstance(ce_loss, float) else ce_loss.item() * batch_size
                total_dice_loss += dice_loss * batch_size if isinstance(dice_loss, float) else dice_loss.item() * batch_size
                
                # Update boundary loss tracking if present
                if boundary_loss != 0:
                    total_boundary_loss += boundary_loss * batch_size if isinstance(boundary_loss, float) else boundary_loss.item() * batch_size
                
                # Update progress bar with current loss values
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'ce': f"{ce_loss if isinstance(ce_loss, float) else ce_loss.item():.4f}",
                    'dice': f"{dice_loss if isinstance(dice_loss, float) else dice_loss.item():.4f}",
                    'lr': f"{self.get_current_lr():.6f}"
                }
                
                # Add boundary loss to progress bar if available
                if boundary_loss != 0:
                    postfix_dict['bound'] = f"{boundary_loss if isinstance(boundary_loss, float) else boundary_loss.item():.4f}"
                    
                pbar.update(1)
                pbar.set_postfix(postfix_dict)
                
                # Optional logging for step-wise metrics (e.g., TensorBoard)
                if hasattr(self, 'log_metrics') and callable(getattr(self, 'log_metrics')):
                    step = batch_idx + (self.current_epoch * batch_count)
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/ce_loss': ce_loss if isinstance(ce_loss, float) else ce_loss.item(),
                        'train/dice_loss': dice_loss if isinstance(dice_loss, float) else dice_loss.item(),
                        'train/lr': self.get_current_lr()
                    }
                    
                    # Add boundary loss to logging if available
                    if boundary_loss != 0:
                        log_dict['train/boundary_loss'] = boundary_loss if isinstance(boundary_loss, float) else boundary_loss.item()
                        
                    self.log_metrics(log_dict, step)
            
            # Compute average metrics for the epoch
            avg_loss = total_loss / processed_samples
            avg_ce_loss = total_ce_loss / processed_samples
            avg_dice_loss = total_dice_loss / processed_samples
            avg_boundary_loss = total_boundary_loss / processed_samples if total_boundary_loss > 0 else 0.0
            
            # Step the scheduler if it's epoch-based
            if self.scheduler is not None and not self.step_scheduler_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch completion
            log_message = (
                f"Epoch {self.current_epoch} completed: "
                f"Loss: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f}"
            )
            
            # Add boundary loss to log if non-zero
            if avg_boundary_loss > 0:
                log_message += f", Boundary: {avg_boundary_loss:.4f}"
            
            log_message += f", LR: {self.get_current_lr():.6f}"
            self.logger.info(log_message)
            
            # Return average metrics
            result = {
                'loss': avg_loss,
                'ce_loss': avg_ce_loss,
                'dice_loss': avg_dice_loss
            }
            
            # Add boundary loss to results if non-zero
            if avg_boundary_loss > 0:
                result['boundary_loss'] = avg_boundary_loss
            
            return result
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation with memory optimizations."""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        # For evaluation
        all_predictions = []
        all_ground_truths = []
        sequence_names = []
        
        # Process in smaller chunks to avoid memory issues
        max_sequences_per_chunk = 20  # Adjust based on your system's RAM
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for batch_idx, batch in enumerate(pbar):
                try:
                    frames = batch['frames'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    sequence_name = batch['sequence'][0] if 'sequence' in batch else f"sequence_{batch_idx}"
                    
                    with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                        outputs = self.model(frames)
                        losses = self.criterion(outputs, {'masks': masks})
                        total_loss += losses['loss'].item()  # Use 'loss' instead of 'total_loss'
                    
                    # Store predictions for evaluation (but limit memory usage)
                    if len(all_predictions) < max_sequences_per_chunk:
                        all_predictions.append(outputs['pred_masks'][0].cpu())
                        all_ground_truths.append(masks[0].cpu())
                        sequence_names.append(sequence_name)
                    
                    # Visualize only occasionally
                    if self.enable_visualization and self.global_step % (self.visualization_interval * 10) == 0 and batch_idx % 50 == 0:
                        self.visualizer.visualize_sequence(
                            frames=frames[0].cpu(),
                            pred_masks=outputs['pred_masks'][0].cpu(),
                            gt_masks=masks[0].cpu(),
                            sequence_name=f"{sequence_name}_epoch_{self.epoch}_lite",
                            max_frames=2  # Only visualize 2 frames, not the whole sequence
                        )
                    
                    # Clear memory
                    del outputs, frames, masks
                    torch.cuda.empty_cache()
                    
                    # Evaluate and reset if we've accumulated enough sequences
                    if len(all_predictions) >= max_sequences_per_chunk or batch_idx == num_batches - 1:
                        if all_predictions and self.enable_evaluation:
                            eval_metrics = self._evaluate_current_predictions(all_predictions, all_ground_truths, sequence_names)
                            
                            # Clear predictions to free memory
                            all_predictions = []
                            all_ground_truths = []
                            sequence_names = []
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.warning(f"OOM during validation at batch {batch_idx}. Clearing memory and continuing...")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        
                        # Clear accumulated data
                        all_predictions = []
                        all_ground_truths = []
                        sequence_names = []
                        
                        continue
                    else:
                        raise e
        
        # Calculate metrics
        metrics = {'val_loss': total_loss / num_batches}
        
        # Add evaluation metrics if available
        if hasattr(self, 'eval_metrics') and self.eval_metrics:
            for key, value in self.eval_metrics.items():
                metrics[key] = value
        
        return metrics

    def _evaluate_current_predictions(self, predictions, ground_truths, seq_names):
        """Helper method to evaluate accumulated predictions in chunks."""
        if self.enable_evaluation and predictions:
            try:
                eval_results = self.evaluator.evaluate(
                    predictions=predictions,
                    ground_truths=ground_truths,
                    sequence_names=seq_names
                )
                
                # Store global metrics for reporting
                self.eval_metrics = eval_results['global']
                
                # Only log abbreviated results
                self.logger.info("\nPartial Evaluation Results:")
                self.logger.info(f"Global: J&F: {eval_results['global']['J&F']:.4f}, "
                                f"J_mean: {eval_results['global']['J_mean']:.4f}, "
                                f"T_mean: {eval_results['global']['T_mean']:.4f}")
                
                return eval_results['global']
                
            except Exception as e:
                self.logger.error(f"Error during evaluation: {str(e)}")
                return {}
        return {}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        validate_every: int = 1,
        save_every: int = 10
    ):
        """
        Main training loop with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Total number of epochs to train
            validate_every: Frequency of validation in epochs
            save_every: Frequency of checkpointing in epochs
        """
        self.logger.info(f"Starting training from epoch {self.epoch}")
        
        for epoch in range(self.epoch, num_epochs):
            for module in self.model.modules():
                if isinstance(module, TemporalFeatureBank):
                    module.features.clear()
            self.epoch = epoch
            self.current_epoch = epoch  # Add this line to update the property
            
            # Training
            train_loss = self.train_epoch(train_loader)
            #self.logger.info(f"Epoch {epoch} training loss: {train_loss['loss']:.4f}")

            
            # Validation
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_losses = self.validate(val_loader)
                self.logger.info(
                    f"Epoch {epoch} validation: " +
                    " ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
                )
                
                # Save checkpoint if it's the best model so far
                if val_losses['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['val_loss']
                    self.save_checkpoint(val_losses, name='model_best')
                
                # Regular checkpoint saving
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(val_losses, name=f'model_epoch_{epoch}')
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.logger.info("Training completed!")

    def compute_binary_metrics(predictions, ground_truths):
        """
        Compute metrics for binary video segmentation evaluation.
        
        This function calculates IoU (Intersection over Union), precision, recall,
        and F1 score for binary segmentation masks. Each metric is calculated per frame
        and then averaged across all frames in all sequences.
        
        Args:
            predictions: List of prediction tensors, each with shape [T, 1, H, W]
            ground_truths: List of ground truth tensors, each with shape [T, H, W]
            
        Returns:
            Dictionary containing averaged metrics:
            - iou: Intersection over Union (Jaccard index)
            - precision: Precision (TP / (TP + FP))
            - recall: Recall (TP / (TP + FN))
            - f1: F1 score (2 * precision * recall / (precision + recall))
        """
        # Initialize accumulators
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_frames = 0
        
        # Process each sequence
        for pred, gt in zip(predictions, ground_truths):
            # Convert predictions to binary (threshold at 0.5)
            # Ensure we're working with the correct dimensions
            if pred.dim() == 4:  # [T, 1, H, W]
                binary_pred = (pred.squeeze(1) > 0.5).bool()
            else:  # [T, H, W]
                binary_pred = (pred > 0.5).bool()
            
            # Convert ground truth to binary (values > 0 are foreground)
            binary_gt = (gt > 0).bool()
            
            # Process each frame in the sequence
            for t in range(binary_pred.shape[0]):
                # Calculate true positives, false positives, false negatives
                tp = (binary_pred[t] & binary_gt[t]).sum().float()
                fp = (binary_pred[t] & ~binary_gt[t]).sum().float()
                fn = (~binary_pred[t] & binary_gt[t]).sum().float()
                
                # Calculate metrics (add small epsilon to avoid division by zero)
                epsilon = 1e-8
                intersection = tp
                union = tp + fp + fn
                
                iou = intersection / (union + epsilon)
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2 * precision * recall / (precision + recall + epsilon)
                
                # Accumulate metrics
                total_iou += iou.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_f1 += f1.item()
                total_frames += 1
                
                # Log individual frame metrics for debugging (optional)
                # logger.debug(f"Frame {total_frames}: IoU={iou.item():.4f}, F1={f1.item():.4f}")
        
        # Handle empty case
        if total_frames == 0:
            logger.warning("No frames were processed during metric calculation")
            return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate and return averages
        return {
            'iou': total_iou / total_frames,
            'precision': total_precision / total_frames,
            'recall': total_recall / total_frames,
            'f1': total_f1 / total_frames
        }
    def evaluate(self, val_loader, visualize=False):
        """
        Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Import visualizer if visualization is requested
        if visualize:
            from utils.visualization import BinarySegmentationVisualizer
            visualizer = BinarySegmentationVisualizer(
                save_dir=os.path.join(self.checkpoint_dir, 'visualizations')
            )
        
        self.model.eval()
        
        # Initialize tracking variables
        total_loss = 0.0
        batch_sizes = 0
        all_predictions = []
        all_ground_truths = []
        all_frames = []
        all_seq_names = []
        
        # Evaluate without gradient computation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get data from batch
                frames = batch['frames'].to(self.device)
                masks = batch['masks'].to(self.device)
                sequence_names = batch.get('sequence_name', [f"seq_{i}" for i in range(frames.shape[0])])
                
                # Forward pass
                outputs = self.model(frames)
                
                # Compute loss
                loss_dict = self.criterion(outputs, {'masks': masks})
                loss = loss_dict.get('loss', loss_dict.get('total_loss', 0.0))
                
                # Track metrics
                batch_size = frames.shape[0]
                total_loss += loss.item() * batch_size
                batch_sizes += batch_size
                
                # Store predictions and ground truth for metrics
                pred_masks = outputs['pred_masks']  # [B, T, 1, H, W]
                
                # Store results for each sequence
                for i in range(frames.shape[0]):
                    all_predictions.append(pred_masks[i])
                    all_ground_truths.append(masks[i])
                    all_frames.append(frames[i])
                    all_seq_names.append(sequence_names[i])
        
        # Calculate average loss
        avg_loss = total_loss / batch_sizes
        
        # Compute segmentation metrics
        metrics = self.compute_binary_metrics(all_predictions, all_ground_truths)
        metrics['loss'] = avg_loss
        
        # Generate visualizations if requested
        if visualize:
            num_viz = min(5, len(all_frames))
            for i in range(num_viz):
                viz_path = os.path.join(
                    visualizer.save_dir, 
                    f"{all_seq_names[i]}_epoch{getattr(self, 'current_epoch', 0)}.png"
                )
                visualizer.visualize_predictions(
                    all_frames[i].unsqueeze(0),
                    all_predictions[i].unsqueeze(0),
                    all_ground_truths[i].unsqueeze(0),
                    save_path=viz_path
                )
        
        # Log metrics
        self.logger.info(
            f"Validation - Loss: {avg_loss:.4f}, "
            f"IoU: {metrics['iou']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )
        
        return metrics
            