from .core_base import Core
import torch
from .metrics import Metrics
from .validation import Validation
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class Training(Core):
    """Training class for model training"""

    def __init__(self, config, 
                 package_name='x_core_ai.src',
                 df_train=None,
                 df_val=None,
                 train_dataloader=None,
                 val_dataloader=None):
        super().__init__(config, package_name)
        
        # Setup training dataset/dataloader
        if df_train is None and train_dataloader is None:
            df = self._get_dataframe(self.config['data_name'], **self.config['data_kwargs'])
            train_dataset = self.create_dataset(df.df_train, train=True)
            self.train_dataloader = self.create_dataloader(train_dataset, train=True)
        elif df_train is not None:
            train_dataset = self.create_dataset(df_train, train=True)
            self.train_dataloader = self.create_dataloader(train_dataset, train=True)
        elif train_dataloader is not None:
            self.train_dataloader = train_dataloader
        
        # Setup validation class for evaluation
        self.validation = Validation(
            config=config,
            package_name=package_name,
            df_val=df_val,
            val_dataloader=val_dataloader
        )
        
        print(f"Training dataloader: {len(self.train_dataloader)} batches")
        print(f"Validation dataloader: {len(self.validation.val_dataloader)} batches")
        
        # Setup model, optimizer, loss, and scheduler
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.setup_scheduler()
        
        # Track best model and metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.early_stop_patience = self.config.get('trainer_kwargs', {}).get('early_stopping_kwargs', {}).get('patience', 20)
        self.early_stop_min_delta = self.config.get('trainer_kwargs', {}).get('early_stopping_kwargs', {}).get('min_delta', 0.001)
        
        # Get save directory for checkpoints
        save_path = self.config.get('model_checkpoint_path', 'checkpoints')
        project_name = self.config.get('project_name', 'model')
        
        # Fix path if it contains raw string literal prefix
        if isinstance(save_path, str):
            if save_path.startswith('r"') or save_path.startswith("r'"):
                save_path = save_path[2:-1]  # Remove r" and closing quote
        
        self.save_dir = os.path.join(save_path, project_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def setup_model(self):
        """Setup model for training"""
        self.model_generator()
        self.model_to_device()
        self.model.train()
        print("Model setup complete")

    def setup_optimizer(self):
        """Setup optimizer based on config"""
        optimizer_name = self.config.get('trainer_kwargs', {}).get('optimizer', 'adam').lower()
        optimizer_kwargs = self.config.get('trainer_kwargs', {}).get('optimizer_kwargs', {})
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_kwargs.get('lr', 0.001),
                weight_decay=optimizer_kwargs.get('weight_decay', 0.01),
                betas=optimizer_kwargs.get('betas', (0.9, 0.999)),
                eps=optimizer_kwargs.get('eps', 1e-8)
            )
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_kwargs.get('lr', 0.001),
                weight_decay=optimizer_kwargs.get('weight_decay', 0.01),
                betas=optimizer_kwargs.get('betas', (0.9, 0.999)),
                eps=optimizer_kwargs.get('eps', 1e-8)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_kwargs.get('lr', 0.01),
                momentum=optimizer_kwargs.get('momentum', 0.9),
                weight_decay=optimizer_kwargs.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        print(f"Optimizer setup: {optimizer_name}")

    def setup_loss_functions(self):
        """Setup loss functions based on config"""
        self.loss_kwargs = self.config.get('loss_kwargs', {})
        self.loss_fns = {}
        
        for task, task_config in self.loss_kwargs.items():
            loss_fn_name = task_config.get('loss_fn', 'CrossEntropyLoss')
            
            if loss_fn_name == 'CrossEntropyLoss':
                ignore_index = task_config.get('ignore_index', -100)
                self.loss_fns[task] = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            elif loss_fn_name == 'BCEWithLogitsLoss':
                self.loss_fns[task] = torch.nn.BCEWithLogitsLoss()
            elif loss_fn_name == 'MSELoss':
                self.loss_fns[task] = torch.nn.MSELoss()
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn_name}")
        
        print(f"Loss functions setup: {', '.join(self.loss_fns.keys())}")

    def setup_scheduler(self):
        """Setup learning rate scheduler based on config"""
        scheduler_name = self.config.get('trainer_kwargs', {}).get('scheduler', None)
        if not scheduler_name:
            self.scheduler = None
            return
        
        scheduler_kwargs = self.config.get('trainer_kwargs', {}).get('scheduler_kwargs', {})
        
        if scheduler_name.lower() == 'cosine':
            total_epochs = self.config.get('trainer_kwargs', {}).get('epochs', 100)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=scheduler_kwargs.get('eta_min', 0)
            )
        elif scheduler_name.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_kwargs.get('mode', 'min'),
                factor=scheduler_kwargs.get('factor', 0.1),
                patience=scheduler_kwargs.get('patience', 10),
                verbose=scheduler_kwargs.get('verbose', True)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        print(f"Scheduler setup: {scheduler_name}")

    def get_inputs_targets(self, batch, drop_keys=[]):
        """Get inputs and targets from batch"""
        inputs = batch['inputs']
        targets = batch['targets']  
        # to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = {k: v.to(self.device) for k, v in targets.items()}
        if drop_keys:
            for key in drop_keys:
                if key in inputs:
                    inputs.pop(key)
        return inputs, targets

    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_dataloader)
        
        start_time = time.time()
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Forward pass
            inputs, targets = self.get_inputs_targets(batch)
            self.optimizer.zero_grad()
            
            outputs = self.model(**inputs)
            
            # Calculate loss for each task
            batch_loss = 0
            for task, loss_fn in self.loss_fns.items():
                if task in outputs and task in targets:
                    # For CrossEntropyLoss, reshape if needed
                    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                        logits = outputs[task]
                        if len(logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
                            task_loss = loss_fn(
                                logits.view(-1, logits.size(-1)),
                                targets[task].view(-1)
                            )
                        else:
                            task_loss = loss_fn(logits, targets[task])
                    else:
                        task_loss = loss_fn(outputs[task], targets[task])
                    
                    # Apply task weight if specified
                    task_weight = self.loss_kwargs[task].get('weight', 1.0)
                    batch_loss += task_weight * task_loss
            
            # Backward pass
            batch_loss.backward()
            self.optimizer.step()
            
            # Update tracking
            epoch_loss += batch_loss.item()
            
            # Print progress
            if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {batch_loss.item():.4f} | "
                      f"Time: {elapsed:.2f}s")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate(self, epoch):
        """Run validation using the Validation class"""
        # Since both trainers will share the model, ensure it's using the latest weights
        self.validation.model = self.model
        self.model.eval()  # Set to evaluation mode
        
        # Run validation
        val_metrics = self.validation.get_validation_metrics()
        
        # Calculate and add validation loss
        val_loss = self.calculate_validation_loss()
        val_metrics['val_loss'] = val_loss
        
        # Print validation results
        print(f"Validation Epoch {epoch} | Val Loss: {val_loss:.4f}")
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        print(f"Metrics: {metric_str}")
        
        return val_metrics
    
    def calculate_validation_loss(self):
        """Calculate validation loss separately"""
        val_loss = 0
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.validation.val_dataloader:
                inputs, targets = self.get_inputs_targets(batch)
                outputs = self.model(**inputs)
                
                # Calculate batch loss
                batch_loss = 0
                for task, loss_fn in self.loss_fns.items():
                    if task in outputs and task in targets:
                        # For CrossEntropyLoss, reshape if needed
                        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                            logits = outputs[task]
                            if len(logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
                                task_loss = loss_fn(
                                    logits.view(-1, logits.size(-1)),
                                    targets[task].view(-1)
                                )
                            else:
                                task_loss = loss_fn(logits, targets[task])
                        else:
                            task_loss = loss_fn(outputs[task], targets[task])
                        
                        task_weight = self.loss_kwargs[task].get('weight', 1.0)
                        batch_loss += task_weight * task_loss
                
                val_loss += batch_loss.item()
        
        return val_loss / len(self.validation.val_dataloader)

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
        
        return checkpoint_path

    def train(self):
        """Train the model for the specified number of epochs"""
        num_epochs = self.config.get('trainer_kwargs', {}).get('epochs', 100)
        save_every = self.config.get('trainer_kwargs', {}).get('save_every', 10)
        
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*20} Epoch {epoch}/{num_epochs} {'='*20}")
            
            # Train one epoch
            train_loss = self.train_one_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.early_stop_counter = 0
                print(f"New best model! Val Loss: {val_loss:.4f}")
            else:
                self.early_stop_counter += 1
                print(f"Not improved for {self.early_stop_counter} epochs. Best: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            
            # Save checkpoint if needed
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                # Print current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")
            
            # Early stopping check
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print(f"\nTraining completed. Best model at epoch {self.best_epoch} with val_loss: {self.best_val_loss:.4f}")
        return self.best_val_loss

    def get_validation_metrics(self):
        """Run validation and return metrics"""
        return self.validation.get_validation_metrics()
