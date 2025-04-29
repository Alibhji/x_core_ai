import os
import importlib
import torch
import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import x_core_ai
from x_core_ai.src.models.tokenizer_zoo import TokenizerZoo
from typing import Dict, Tuple, Any, Optional
from x_core_ai.src.utils.model_utils import analyze_model_size, load_weights
from .utils import custom_collate_fn



class Core:
    def __init__(self, config, package_name='x_core_ai.src'):
        """Base Core class with common functionality for all ML pipeline stages"""
        self.config = config
        self.setup_device()
        self.setup_package(package_name)
        self.dataloaders = {}
        # self.tokenizer = self.get_tokenizer()
        self.tokenizer = TokenizerZoo.get_tokenizer(self.config['tokenizer_name'])

    def setup_device(self):
        """Setup compute device (CPU/GPU)"""
        gpus = self.config.get('gpus', list(range(torch.cuda.device_count())))
        primary_gpu = gpus[0] if gpus else 0
        self.device = torch.device(f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu")
        if self.config.get('distributed', False):
            self.device = f"cuda:{primary_gpu}"

    def get_tokenizer(self):
        """Get tokenizer from registry"""
        return x_core_ai.src.tokenizers.registry.get_tokenizer(self.config['tokenizer_name'])
            
    def setup_package(self, package_name):
        """Setup package imports and paths"""
        spec = importlib.util.find_spec(package_name)
        self.root_package_path = os.path.dirname(os.path.abspath(spec.origin))
        
        # Import necessary components based on config
        if self.config.get('dataset_name'):
            self.package_importer('datasets/pool_auto', self.config['dataset_name'])
        if self.config.get('model_name'):
            self.package_importer('models/pool_auto', self.config['model_name'])
        if self.config.get('data_name'):
            self.package_importer('dataframes/pool_auto', self.config['data_name'])
    
    def package_importer(self, relative_path_inside_package, module_list):
        """Import modules from the package dynamically"""
        module_list = module_list if isinstance(module_list, list) else [module_list]
        relative_module_path = os.path.join(*self.root_package_path.split(os.sep)[-2:], 
                                          *relative_path_inside_package.split('/')).replace(os.sep, '.')
        for file in module_list:
            module_name = f"{relative_module_path}.{file}"
            print(f"Importing module: {module_name}")
            importlib.import_module(module_name)
            
    def model_generator(self):
        """Generate and initialize model"""
        self.model = self.get_model(self.config["model_name"], 
                               **self.config["model_kwargs"])
        
        # Load weights if specified in config
        weights_path = self.config.get('checkpoint_weight_path')
        if weights_path is not None:
            self.load_model_weights(self.model, weights_path)
            print(f"Loaded model weights from {weights_path}")
        
        # Analyze and print model statistics
        analyze_model_size(self.model)
    
    def model_to_device(self):
        """Move model to appropriate device and setup distributed if needed"""
        gpus = self.config.get('gpus', [0])
        if self.config.get("distributed", False):
            print(f' >>> [DDP Training] --> Rank: {self.config["dist_kwargs"]["local_rank"]} ')
            self.device = torch.device(f"cuda:{self.config['dist_kwargs']['local_rank']}" 
                                     if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            find_unused_parameters = self.config.get('find_unused_parameters', False)
            self.model = DDP(self.model, 
                           device_ids=[self.config['dist_kwargs']['local_rank']],
                           find_unused_parameters=find_unused_parameters,
                           output_device=self.config['dist_kwargs']['local_rank'])
        elif len(gpus) > 1:
            print(f' >>> [DP Training] --> with {len(gpus)} GPUs')
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model)
        else:
            self.model.to(self.device)
            print(f' >>> [Single GPU Training]')
            
    def get_model(self, name, *args, **kwargs):
        """Get model from registry"""
        return x_core_ai.src.models.registry.get_model(name,*args, **kwargs)
    
    def get_dataset(self, name, *args, **kwargs):
        """Get dataset from registry"""
        return x_core_ai.src.datasets.registry.get_dataset(name, *args, **kwargs)
    
    def _get_dataframe(self, name, *args, **kwargs):
        """Get dataframe from registry"""
        return x_core_ai.src.dataframes.registry.get_dataframe(name, *args, **kwargs)
        
    def fix_batch_format(self, batch):
        """Fix batch format for model input"""
        try:
            primary_key = batch.pop('primary_key') 
        except:
            batch = self.custom_collate_fn(batch)
            primary_key = batch.pop('primary_key', None)
        return primary_key, batch
    
    def create_dataframes(self):
        """Get dataframes from registry based on config"""
        if not self.config.get('data_name'):
            return None, None, None
            
        if isinstance(self.config['data_name'], list):
            dataframes = []
            for i, name in enumerate(self.config['data_name']):
                kwargs_name = 'data_kwargs'
                str_ = f'_{i}' if i != 0 else ''
                kwargs_name += str_
                dataframes.append(self._get_dataframe(name, **self.config[kwargs_name]))
            df_train = pd.concat([df_.df_train for df_ in dataframes])
            df_val = pd.concat([df_.df_val for df_ in dataframes])
            df_test = pd.concat([df_.df_test for df_ in dataframes])
        else:
            dataframe = self._get_dataframe(self.config['data_name'], **self.config['data_kwargs'])
            df_train = dataframe.df_train
            df_test = dataframe.df_test
            df_val = getattr(dataframe, 'df_val', df_test)  # Use df_test as fallback if df_val doesn't exist
            
        return df_train, df_val, df_test
    
    def create_dataset(self, df=None, train=True):
        """Create dataset from dataframe"""
        # Get dataframe if not provided
        if df is None:
            df_train, df_val, df_test = self.create_dataframes()
            df = df_train if train else df_val
            
        if df is None:
            return None
            
        # Create dataset
        dataset = self.get_dataset(
            self.config['dataset_name'],
            df=df,
            train=train,
            **self.config['dataset_kwargs'])
            
        
        return dataset
    
    def create_dataloader(self, dataset, train=True, batch_size=None, shuffle=None):
        """Create dataloader from dataset"""
        # Get dataloader parameters
        dataloader_kwargs = self.config[f'{"train" if train else "val"}_dataloader_kwargs'].copy()

            
        # Create sampler for distributed training
        sampler = None
        if self.config.get('distributed', False):
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config['dist_kwargs']['world_size'],
                rank=self.config['dist_kwargs']['local_rank'],
                shuffle=dataloader_kwargs.get('shuffle', True)
            )
            dataloader_kwargs.pop('shuffle', None)  # Remove shuffle if using sampler
            
        # Create dataloader
        dataloader_kwargs = self.config['train_dataloader_kwargs'] if train else self.config['val_dataloader_kwargs']
        shuffle = dataloader_kwargs.pop('shuffle', True)
        dataloader = DataLoader(
            dataset, 
            sampler=sampler,
            shuffle= shuffle if (sampler is None or train) else False, # if sampler is not None, then shuffle is False or if val then False
            **dataloader_kwargs,
            collate_fn= custom_collate_fn)
        
        return dataloader
    
    # def get_train_dataloader(self):
    #     """Get training dataloader"""
    #     dataframe = self.get_dataframe(self.config['data_name'], **self.config['data_kwargs']) 
    #     dataset = self.get_dataset(self.config['dataset_name'], 
    #                                   df=dataframe.df_train, 
    #                                   train=True, 
    #                                   **self.config['dataset_kwargs'])

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, 
    #         **self.config['trainer_kwargs']['train_dataloader_kwargs'],
    #         collate_fn= custom_collate_fn)
    #     return dataloader
    

    # def get_val_dataloader(self):
    #     """Get validation dataloader"""
    #     dataframe = self.get_dataframe(self.config['data_name'], **self.config['data_kwargs']) 
    #     dataset = self.get_dataset(self.config['dataset_name'], 
    #                                   df=dataframe.df_val, 
    #                                   train=False, 
    #                                   **self.config['dataset_kwargs'])

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, 
    #         **self.config['trainer_kwargs']['val_dataloader_kwargs'],
    #         collate_fn= custom_collate_fn)
    #     return dataloader
    
    def get_train_dataloader(self):
        """Get training dataloader"""
        return self.create_dataloader(train=True)
    
    def get_val_dataloader(self):
        """Get validation dataloader"""
        return self.create_dataloader(train=False)
    
    # @staticmethod
    # def custom_collate_fn(batch):
    #     """Custom collate function for DataLoader"""
    #     collated_batch = {}
    #     # Get all keys in the dataset
    #     keys = batch[0].keys()

    #     for key in keys:
    #         values = [sample[key] for sample in batch]

    #         if isinstance(values[0], np.ndarray):
    #             try:
    #                 collated_batch[key] = torch.as_tensor(np.stack(values))
    #             except:
    #                 print('issue in stacking arrays')

    #         elif isinstance(values[0], (int, float)):  # Handle numerical values
    #             collated_batch[key] = torch.tensor(values)
                
    #         elif isinstance(values[0], (torch.Tensor, np.ndarray)):
    #             values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in values]
    #             collated_batch[key] = torch.stack(values)  # Stack tensors

    #         elif isinstance(values[0], (str, list)):  # Handle strings/lists
    #             collated_batch[key] = values  # Keep them as lists
                
    #         elif isinstance(values[0], dict):
    #             collated_batch[key] = Core.custom_collate_fn(values)
    #         else:
    #             raise TypeError(f"Unsupported data type for key '{key}': {type(values[0])}")

    #     return collated_batch
    
    @staticmethod
    def load_model_weights(model, weights_path):
        """Load model weights from checkpoint"""
        if not os.path.exists(weights_path):
            print(f"Weights file not found: {weights_path}")
            return
            
        # Add datetime.date to safe globals before attempting to load
        from datetime import date
        torch.serialization.add_safe_globals([date])
            
        try:
            # First try loading with weights_only=True (default in PyTorch 2.6+)
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f"Initial load failed, attempting to load with weights_only=False: {str(e)}")
            try:
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load weights: {str(e)}")
                return
            
        # Extract model state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Handle module. prefix from DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        # Load state dict with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Some weights were not initialized from the checkpoint: {missing_keys}")
            print("You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
        if unexpected_keys:
            print(f"Some weights from the checkpoint were not used: {unexpected_keys}")
            
        print(f"Loaded model weights from {weights_path}")
    
    @staticmethod
    def cleanup():
        """Destroy the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group() 