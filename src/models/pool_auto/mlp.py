import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from torch.utils.data import DataLoader, Dataset
import math
from ..registry import register_model

# Gated Cross-Attention Network (GCAN)
@register_model("mlp")
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        
        # Initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Initialize activation function
        self.activation = nn.ReLU() if activation == "relu" else nn.Sigmoid()   
        
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x    
