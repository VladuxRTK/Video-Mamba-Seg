import torch
import torch.nn as nn
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import cv2
import math
import time
import copy
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
        from losses.segmentation import BinarySegmentationLoss
        self.criterion = BinarySegmentationLoss(
            ce_weight=config['losses']['ce_weight'],
            dice_weight=config['losses']['dice_weight'],
            boundary_weight=config['losses'].get('boundary_weight', 0.0)
        )
        
        # Training settings - using old PyTorch syntax
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_value = config['training'].get('grad_clip_value', 0.0)
        self.step_scheduler_batch = step_scheduler_batch
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
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
                
                # Forward pass with mixed precision - using old PyTorch syntax
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
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
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{get_item_safely(loss):.4f}",
                    'dice': f"{get_item_safely(dice_loss):.4f}",
                    'lr': f"{self.get_current_lr():.6f}"
                })
                
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
                
                # Track loss
                total_loss += loss.item() * frames.shape[0]
                
                # Store predictions for evaluation (detach and move to CPU to save memory)
                if self.enable_evaluation:
                    # Store only the first item in batch to save memory
                    all_predictions.append(outputs['pred_masks'][0].cpu())
                    all_ground_truths.append(masks[0].cpu())
                    all_sequences.append(sequence[0] if isinstance(sequence, list) else sequence)
                
                # Visualize predictions occasionally
                if self.enable_visualization and batch_idx % self.visualization_interval == 0:
                    self._visualize_prediction(
                        frames=frames[0].cpu(),
                        pred_masks=outputs['pred_masks'][0].cpu(),
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        validate_every: int = 1,
        save_every: int = 10,
        patience: int = 15  # Early stopping patience
    ):
        """Main training loop with early stopping."""
        self.logger.info(f"Starting training from epoch {self.epoch}")
        
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
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                
                self.logger.info(
                    f"Epoch {epoch} validation - Loss: {val_loss:.4f}"
                )
                
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
        
        self.logger.info("Training completed!")
        
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
        """
        Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary of evaluation metrics
        """
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
                
                # Track loss
                total_loss += loss.item() * frames.shape[0]
                
                # Store predictions for metrics calculation
                all_predictions.append(outputs['pred_masks'][0].cpu())
                all_ground_truths.append(masks[0].cpu())
                all_sequences.append(sequence[0] if isinstance(sequence, list) else sequence)
                
                # Create visualizations for specific batches
                if visualize and batch_idx % self.visualization_interval == 0:
                    self._visualize_prediction(
                        frames=frames[0].cpu(),
                        pred_masks=outputs['pred_masks'][0].cpu(),
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