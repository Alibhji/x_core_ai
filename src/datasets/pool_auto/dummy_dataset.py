from ..registry import register_dataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Union

@register_dataset("dummy_dataset")
class DummyDataset(Dataset):
    def __init__(self, 
                 num_samples: int,
                 feature_dim: int,
                 sequence_length: int,
                 random_seed: int = 42):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.data = self._generate_data()
    
    def _generate_data(self):
        np.random.seed(self.random_seed)
        data = np.random.randn(self.num_samples, self.sequence_length, self.feature_dim)
        return torch.tensor(data, dtype=torch.float)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]   
    
    def zero_padding(self, data):
        if data.shape[0] < self.sequence_length:
            padding = torch.zeros(self.sequence_length - data.shape[0], data.shape[1], data.shape[2])
            data = torch.cat([data, padding], dim=0)
        return data
    
    def get_data(self):
        return self.data
 

