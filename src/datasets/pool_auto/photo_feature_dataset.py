from ..registry import register_dataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Union

@register_dataset("photo_feature_dataset")
class PhotoFeatureDataset(Dataset):
    def __init__(self, 
                 df=None,
                 photo_feature_sequence_length=40,
                 photo_feature_dim=768,
                 target_name="cost_target",
                 target_type="regression",
                 target_range=[0, 2000],
                 train=True,
                 train_split=0.8,
                 val_split=0.1,
                 test_split=0.1,
                 storage_path=None,
                 **kwargs
     ):
        """
        Photo Feature Dataset
        Args:
            df: Input dataframe with photo_features and target columns
            photo_feature_sequence_length: Length of photo feature sequence
            photo_feature_dim: Dimension of photo features
            target_name: Name of the target column
            target_type: Type of target (regression or classification)
            target_range: Range of target values
            train: Whether this is a training dataset
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            storage_path: Path to storage directory for parquet files
        """
        self.photo_feature_sequence_length = photo_feature_sequence_length
        self.photo_feature_dim = photo_feature_dim
        self.target_name = target_name
        self.target_type = target_type
        self.target_range = target_range
        self.is_train = train
        
        # Set the data based on input method
        if df is not None:
            # Use provided dataframe directly
            self.data = df
        elif storage_path and os.path.exists(storage_path):
            # Load from parquet files
            if train:
                parquet_path = f"{storage_path}/train.parquet"
            else:
                parquet_path = f"{storage_path}/val.parquet"
                
            if os.path.exists(parquet_path):
                self.data = pd.read_parquet(parquet_path)
            else:
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        else:
            raise ValueError("Either df or valid storage_path must be provided")
        
        # Apply preprocessing if needed
        if 'photo_features' in self.data.columns:
            # Data already contains processed features
            self.features = self.data['photo_features'].tolist()
            self.targets = self.data[self.target_name].tolist()
        else:
            raise ValueError(f"DataFrame must contain 'photo_features' and '{target_name}' columns")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get features and ensure they're tensors
        features = self.features[idx]
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
            
        # Zero pad if needed
        if features.shape[0] < self.photo_feature_sequence_length:
            features = self.zero_padding(features)
        elif features.shape[0] > self.photo_feature_sequence_length:
            # Truncate if too long
            features = features[:self.photo_feature_sequence_length]
            
        # Get target and ensure it's a tensor
        target = self.targets[idx]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float)
            
        # Return as dictionary for flexibility
        # Use photo_feat key for GCAN compatibility
        return {
            'photo_feat': features,
            self.target_name: target
        }
        
    def zero_padding(self, photo_feature):
        """Add zero padding to features that are too short"""
        if not isinstance(photo_feature, torch.Tensor):
            photo_feature = torch.tensor(photo_feature, dtype=torch.float)
            
        if photo_feature.shape[0] < self.photo_feature_sequence_length:
            padding = torch.zeros(
                self.photo_feature_sequence_length - photo_feature.shape[0], 
                self.photo_feature_dim,
                dtype=photo_feature.dtype,
                device=photo_feature.device
            )
            photo_feature = torch.cat([photo_feature, padding])
            
        return photo_feature




