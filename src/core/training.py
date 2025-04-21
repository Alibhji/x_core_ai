import os
import time
import torch
import numpy as np
from .validation import Validation

# Import MLflow integration from experiment_tracker
try:
    from ..experiment_tracker import integrate_tracker_with_core, track_training_epoch
    _has_mlflow = True
except ImportError:
    _has_mlflow = False

class Training(Validation):
    def __init__(self, config, package_name='x_core_ai.src'):
        """Training class for model training"""
        super().__init__(config, package_name)
        self.setup_training()
        
        # Setup MLflow tracking if enabled in config
        self.tracker = None
        if _has_mlflow and self.config.get("experiment_tracking", {}).get("enable", False):
            self.tracker = integrate_tracker_with_core(self)
        
    def setup_training(self):
        """Setup training components"""
        # Set model to training mode
        self.model.train()
        
        # Setup optimizer
        self.optimizer = self.get_optimizer()
        
        # Setup scheduler
        self.scheduler = self.get_scheduler()
        
        # Setup loss function
        self.loss_fn = self.get_loss_function()
        
        # Setup training state
        self.current_epoch = 0
        self.best_val_metric = float('inf')  # Lower is better by default
        self.early_stop_count = 0
        self.train_losses = []
        self.val_metrics = {}
        
    def get_optimizer(self):
        """Get optimizer from config"""
        optimizer_name = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        optimizer_kwargs = self.config.get('optimizer_kwargs', {})
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = lr
        if 'weight_decay' not in optimizer_kwargs:
            optimizer_kwargs['weight_decay'] = weight_decay
            
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
        elif optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
    def get_scheduler(self):
        """Get learning rate scheduler from config"""
        if not self.config.get('scheduler_kwargs'):
            return None
            
        scheduler_type = self.config.get('scheduler', 'cosine')
        scheduler_kwargs = self.config.get('scheduler_kwargs', {})
        
        if scheduler_type.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100),
                **{k: v for k, v in scheduler_kwargs.items() if k != 'T_max'}
            )
        elif scheduler_type.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **scheduler_kwargs
            )
        elif scheduler_type.lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_kwargs
            )
        else:
            return None
            
    def get_loss_function(self):
        """Get loss function from config"""
        loss_name = self.config.get('loss', 'mse')
        
        if loss_name.lower() == 'mse':
            return torch.nn.MSELoss()
        elif loss_name.lower() == 'mae':
            return torch.nn.L1Loss()
        elif loss_name.lower() == 'bce':
            return torch.nn.BCELoss()
        elif loss_name.lower() == 'ce':
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
            
    def train_epoch(self, train_dataloader):
        """
        Train for one epoch
        Args:
            train_dataloader: DataLoader with training data
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Check if using GCAN
        is_gcan = self.config.get('model_name') == 'gated_cross_attention'
        target_name = self.config.get('target_name', 'cost_target')
        
        for batch in train_dataloader:
            # Extract inputs and targets
            if isinstance(batch, dict):
                if target_name in batch:
                    targets = batch[target_name].to(self.device)
                    
                    # For GCAN, use photo_feat
                    if is_gcan and 'photo_feat' in batch:
                        inputs = batch['photo_feat'].to(self.device)
                    else:
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items() if k != target_name}
                else:
                    # Try common fallbacks
                    targets = batch.get('target')
                    if targets is not None:
                        targets = targets.to(self.device)
                    
                    if 'photo_feat' in batch:
                        inputs = batch['photo_feat'].to(self.device)
                    else:
                        # Use first non-target key as input
                        for k, v in batch.items():
                            if k != 'target' and isinstance(v, torch.Tensor):
                                inputs = v.to(self.device)
                                break
            else:
                # Handle tuple of (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if is_gcan and isinstance(inputs, torch.Tensor):
                outputs = self.model(photo_feat=inputs)
            else:
                outputs = self.model(inputs)
            
            # Calculate loss
            if isinstance(outputs, dict):
                if target_name in outputs:
                    outputs = outputs[target_name]
                else:
                    outputs = next(iter(outputs.values()))
            
            # Ensure targets have the same dtype as outputs to avoid dtype mismatch
            if isinstance(targets, torch.Tensor) and isinstance(outputs, torch.Tensor):
                if outputs.dtype != targets.dtype:
                    # For most loss functions, float is expected
                    if self.config.get('loss', 'mse').lower() in ['mse', 'mae', 'bce']:
                        # If using regression losses, convert both to float
                        outputs = outputs.float()
                        targets = targets.float()
                    elif self.config.get('loss', 'mse').lower() in ['ce']:
                        # For CrossEntropyLoss, targets should be Long
                        if outputs.dtype == torch.float32:
                            targets = targets.long()
                    else:
                        # Default case: convert targets to match outputs
                        targets = targets.to(dtype=outputs.dtype)
                
                # Ensure target has the same shape as outputs for loss calculation
                # This fixes the shape mismatch warning
                if outputs.dim() > targets.dim():
                    if outputs.shape[0] == targets.shape[0]:
                        # If batch dimension matches but target is 1D, reshape to match output dimensions
                        targets = targets.reshape(targets.shape[0], *([1] * (outputs.dim() - 1)))
                        # Now expand to match exact output shape
                        targets = targets.expand_as(outputs)
                elif outputs.dim() < targets.dim():
                    # If output has fewer dimensions, squeeze extra dimensions from targets
                    targets = targets.squeeze()
                    # If we still have mismatch, try to reshape targets to match output shape
                    if outputs.shape != targets.shape:
                        try:
                            targets = targets.view(outputs.shape)
                        except:
                            print(f"Warning: Unable to reshape targets {targets.shape} to match outputs {outputs.shape}")
                
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update loss
            epoch_loss += loss.item()
            batch_count += 1
        
        # Calculate average loss
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        return avg_loss
        
    def train(self, train_dataloader=None, val_dataloader=None, progress_callback=None):
        """
        Train the model
        Args:
            train_dataloader: DataLoader with training data (uses default if None)
            val_dataloader: DataLoader with validation data (uses default if None)
            progress_callback: Optional callback for progress updates
        Returns:
            Training history
        """
        # Get dataloaders if not provided
        if train_dataloader is None:
            train_dataloader = self.get_train_dataloader()
        if val_dataloader is None:
            val_dataloader = self.get_val_dataloader()
            
        if train_dataloader is None:
            raise ValueError("No training dataloader available")
        if val_dataloader is None:
            raise ValueError("No validation dataloader available")
        
        # Training parameters
        epochs = self.config.get('epochs', 100)
        early_stopping = self.config.get('early_stopping_kwargs', {}).get('patience', None)
        save_dir = self.config.get('save_dir', 'checkpoints')
        save_every = self.config.get('save_every', 10)
        monitor_metric = self.config.get('monitor_metric', 'loss')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rate': []
        }
        
        # Get metric direction (minimize or maximize)
        minimize_metric = True
        if monitor_metric in ['accuracy', 'r2', 'f1', 'precision', 'recall']:
            minimize_metric = False
            self.best_val_metric = -float('inf')  # For metrics where higher is better
        
        # MLflow configuration
        mlflow_config = self.config.get('experiment_tracking', {})
        mlflow_enabled = mlflow_config.get('enable', False) and _has_mlflow
        mlflow_log_freq = mlflow_config.get('log_freq', 1)
        mlflow_log_model = mlflow_config.get('log_model', True)
        
        # Start MLflow run if enabled
        if mlflow_enabled and self.tracker:
            run_name = mlflow_config.get('run_name', self.config.get('project_name', 'training'))
            self.tracker.start_run(run_name=run_name)
            
            # Log model parameters
            model_params = self.config.get('model_kwargs', {})
            train_params = {
                'epochs': epochs,
                'learning_rate': self.config.get('learning_rate', 0.001),
                'weight_decay': self.config.get('weight_decay', 0),
                'optimizer': self.config.get('optimizer', 'adam'),
                'scheduler': self.config.get('scheduler', ''),
                'loss': self.config.get('loss', 'mse'),
                'batch_size': self.config.get('dataloader_kwargs_train', {}).get('batch_size', 32)
            }
            
            if hasattr(self.tracker, 'log_params'):
                self.tracker.log_params(model_params)
                self.tracker.log_params(train_params)
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Train for one epoch
                train_loss = self.train_epoch(train_dataloader)
                
                # Update learning rate if using step scheduler
                if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                    
                # Validate
                val_metrics = self.evaluate(val_dataloader)
                
                # Update learning rate if using plateau scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_value = val_metrics.get(monitor_metric, train_loss)
                    self.scheduler.step(monitor_value)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_metrics'].append(val_metrics)
                history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # Save checkpoint if needed
                checkpoint_path = None
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch+1}.pth"
                    self.save_checkpoint(checkpoint_path)
                
                # Check for best model
                if monitor_metric == 'loss':
                    current_metric = train_loss
                else:
                    current_metric = val_metrics.get(monitor_metric, float('inf'))
                    
                is_best = False
                if minimize_metric and current_metric < self.best_val_metric:
                    is_best = True
                    self.best_val_metric = current_metric
                    self.early_stop_count = 0
                elif not minimize_metric and current_metric > self.best_val_metric:
                    is_best = True
                    self.best_val_metric = current_metric
                    self.early_stop_count = 0
                else:
                    self.early_stop_count += 1
                    
                best_model_path = None
                if is_best:
                    best_model_path = f"{save_dir}/best_model.pth"
                    self.save_checkpoint(best_model_path)
                    
                # Log metrics to MLflow if enabled
                if mlflow_enabled and self.tracker and epoch % mlflow_log_freq == 0:
                    # Track metrics
                    train_metrics = {'loss': train_loss}
                    
                    # Additional metrics
                    train_metrics.update({
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch_time': epoch_time
                    })
                    
                    # Log metrics
                    track_training_epoch(self, epoch, train_metrics=train_metrics, val_metrics=val_metrics)
                    
                    # Log model checkpoint as artifact if enabled
                    if mlflow_log_model and checkpoint_path and os.path.exists(checkpoint_path):
                        self.tracker.log_artifact(checkpoint_path)
                    
                    # Log best model if it's a new best
                    if is_best and best_model_path and os.path.exists(best_model_path):
                        self.tracker.log_artifact(best_model_path)
                    
                # Print progress
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}s - Loss: {train_loss:.6f}")
                for metric, value in val_metrics.items():
                    print(f"  {metric}: {value:.6f}")
                    
                # Early stopping
                if early_stopping and self.early_stop_count >= early_stopping:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    
                    # Log early stopping to MLflow
                    if mlflow_enabled and self.tracker:
                        self.tracker.log_param("early_stopping_epoch", epoch+1)
                    
                    break
                    
                # Update progress if callback provided
                if progress_callback and callable(progress_callback):
                    progress_callback(epoch + 1, epochs, {
                        'train_loss': train_loss,
                        'val_metrics': val_metrics,
                        'best_metric': self.best_val_metric
                    })
        
        except Exception as e:
            print(f"Error during training: {e}")
            # Log error to MLflow
            if mlflow_enabled and self.tracker:
                self.tracker.log_param("training_error", str(e))
            raise
        
        finally:
            # End MLflow run if enabled
            if mlflow_enabled and self.tracker and hasattr(self.tracker, 'end_run'):
                # Log final metrics before ending run
                if len(history['train_loss']) > 0:
                    final_metrics = {
                        'final_train_loss': history['train_loss'][-1],
                        'best_val_metric': self.best_val_metric,
                        'total_epochs': len(history['train_loss']),
                        'total_training_time': time.time() - start_time
                    }
                    
                    # Add final validation metrics
                    if len(history['val_metrics']) > 0:
                        for metric, value in history['val_metrics'][-1].items():
                            final_metrics[f'final_val_{metric}'] = value
                    
                    # Log metrics
                    self.tracker.log_metrics(final_metrics)
                
                self.tracker.end_run()
        
        return history
    
    def get_train_dataloader(self):
        """Get training dataloader"""
        # Load training dataset from config
        if self.config.get('dataset_name') and self.config.get('dataset_kwargs'):
            df_train = None
            if self.config.get('data_name'):
                dataframe = self.get_dataframe(self.config['data_name'], 
                                              **self.config.get('dataframe_kwargs', {}))
                df_train = dataframe.df_train
                
            dataset = self.get_dataset(self.config['dataset_name'], 
                                      df=df_train, 
                                      train=True, 
                                      **self.config.get('dataset_kwargs', {}))
        else:
            raise ValueError("No training dataset configured")
            
        # Create dataloader
        batch_size = self.config.get('dataloader_kwargs_train', {}).get('batch_size', 32)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            **{k: v for k, v in self.config.get('dataloader_kwargs_train', {}).items() if k != 'batch_size'}
        )
        
        return dataloader
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        # Add scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return False
            
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', {})
        
        print(f"Checkpoint loaded from {path}")
        return True 

    def save_training_visualization(self, history, filename="training_history.png"):
        """
        Save training history visualization to the storage directory.
        
        Args:
            history: Training history dictionary containing metrics
            filename: Name of the output file
        
        Returns:
            Path to the saved visualization file
        """
        import matplotlib.pyplot as plt
        
        # Get storage path from config
        storage_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "storage"
        )
        
        # Use project name and version from config for subfolder structure
        project_name = self.config.get("project_name", "default_project")
        version = self.config.get("version", "v1.0")
        if "experiment_tracking" in self.config:
            project_name = self.config["experiment_tracking"].get("project_name", project_name)
            version = self.config["experiment_tracking"].get("version", version)
        
        # Create directory structure
        save_dir = os.path.join(storage_root, project_name, version, "visualizations")
        os.makedirs(save_dir, exist_ok=True)
        
        # Full path for the output file
        output_path = os.path.join(save_dir, filename)
        
        # Create the visualization
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(history.get('train_loss', []), label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot validation metrics
        plt.subplot(1, 2, 2)
        metrics = self.config.get('metrics', ['loss'])
        for metric in metrics:
            values = []
            for epoch_metrics in history.get('val_metrics', []):
                values.append(epoch_metrics.get(metric, float('nan')))
            plt.plot(values, label=f'Val {metric}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Validation Metrics')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add a suptitle with project and version
        plt.suptitle(f"Training Results - {project_name} ({version})")
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Training visualization saved to {output_path}")
        
        # Log the visualization to MLflow if enabled
        if hasattr(self, 'tracker') and self.tracker and self.config.get("experiment_tracking", {}).get("enable", False):
            self.tracker.log_artifact(output_path)
            print(f"Training visualization logged to MLflow")
        
        return output_path 
    