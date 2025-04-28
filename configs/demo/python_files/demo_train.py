import sys
import os
# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

from x_core_ai.src.core import Training    
from sub_module.utilx.src.config import ConfigLoader
from argparse import ArgumentParser
import torch

def parse_args():
    parser = ArgumentParser(description="Train a model using the x_core_ai framework")
    parser.add_argument("--config", type=str, default="configs/models/multi_task_vit_v1.0.0.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--epochs", type=int, default=None,
                      help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                      help="Override batch size for training")
    parser.add_argument("--lr", type=float, default=None,
                      help="Override learning rate")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to a checkpoint to resume training from")
    parser.add_argument("--save-dir", type=str, default=None,
                      help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, 
                      help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with a predefined configuration")
    return parser.parse_args()

def debug(args):
    """Set debug configuration"""
    args.config = "configs/models/multi_task_vit_v1.0.0.yaml"
    args.epochs = 2
    args.batch_size = 2

def main(args):
    # Load configuration
    config = ConfigLoader.load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.epochs is not None:
        config['trainer_kwargs']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['train_dataloader_kwargs']['batch_size'] = args.batch_size
        config['val_dataloader_kwargs']['batch_size'] = args.batch_size
    
    if args.lr is not None:
        config['trainer_kwargs']['learning_rate'] = args.lr
        if 'optimizer_kwargs' in config['trainer_kwargs']:
            config['trainer_kwargs']['optimizer_kwargs']['lr'] = args.lr
    
    if args.save_dir is not None:
        config['model_checkpoint_path'] = args.save_dir
    
    if args.device is not None:
        if args.device == 'cpu':
            config['gpus'] = []
        elif args.device.startswith('cuda:'):
            gpu_id = int(args.device.split(':')[1])
            config['gpus'] = [gpu_id]
    
    # Initialize training
    print(f"Initializing training with config from {args.config}")
    print(f"Training for {config['trainer_kwargs']['epochs']} epochs")
    print(f"Using batch size: {config['train_dataloader_kwargs']['batch_size']}")
    print(f"Using device: {config['gpus'] if 'gpus' in config and config['gpus'] else 'cpu'}")
    
    # Create trainer instance
    trainer = Training(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint file {args.checkpoint} not found. Starting from scratch.")
    
    # Start training
    try:
        best_val_loss = trainer.train()
        print(f"Training completed successfully! Best validation loss: {best_val_loss:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug(args)
    main(args)