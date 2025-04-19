import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from torch.utils.data import DataLoader, Dataset
import math
from ..registry import register_model

# Gated Cross-Attention Network (GCAN)
@register_model("gated_cross_attention")
class GCAN(nn.Module):
    def __init__(
        self,
        input_shape=(40, 768),
        num_layers=6,
        num_heads=8,
        hidden_dim=2048,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.cost_target_name = kwargs.get('target_name', 'cost_target')
        
        # Use provided parameters or set defaults
        self.input_shape = input_shape 
        self.num_layers = num_layers 
        self.num_heads = num_heads 
        self.hidden_dim = hidden_dim 
        self.dropout = dropout
        
        # Initialize BERT configuration
        config = BertConfig(
            hidden_size=self.input_shape[1],
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            max_position_embeddings=self.input_shape[0],
            position_embedding_type="absolute"
        )
        
        # BERT backbone
        self.bert = BertModel(config)
        
        # Feature gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(self.input_shape[1], self.input_shape[1]),
            nn.LayerNorm(self.input_shape[1]),
            nn.GELU(),
            nn.Linear(self.input_shape[1], self.input_shape[1]),
            nn.Sigmoid()
        )
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.input_shape[1],
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.input_shape[1])
        self.norm2 = nn.LayerNorm(self.input_shape[1])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.input_shape[1], self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.input_shape[1]),
            nn.Dropout(self.dropout)
        )
        
        # Regression head with configurable dimensions
        regressor_hidden_dim = kwargs.get('regressor_hidden_dim', self.hidden_dim)
        regressor_mid_dim = kwargs.get('regressor_mid_dim', 256)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.input_shape[1], regressor_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(regressor_hidden_dim, regressor_mid_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(regressor_mid_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x, eps=1e-6):
        """Create padding mask where True indicates padding"""
        row_sums = torch.abs(x).sum(dim=-1)
        return row_sums < eps
    
    def forward(self,
                photo_feat: torch.Tensor):
        # Create padding mask
        padding_mask = self.create_padding_mask(photo_feat)
        attention_mask = (~padding_mask).float()  # Convert to float for BERT
        
        # BERT processing
        bert_outputs = self.bert(
            inputs_embeds=photo_feat,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = bert_outputs.last_hidden_state
        
        # Feature gating
        gates = self.feature_gate(hidden_states)
        gated_features = hidden_states * gates
        
        # Cross-attention with padding mask
        attn_output, _ = self.cross_attention(
            query=self.norm1(gated_features),
            key=self.norm1(gated_features),
            value=self.norm1(gated_features),
            key_padding_mask=padding_mask
        )
        
        # Residual connection
        hidden_states = gated_features + attn_output
        
        # FFN
        ffn_output = self.ffn(self.norm2(hidden_states))
        hidden_states = hidden_states + ffn_output
        
        # Global average pooling (considering padding mask)
        mask_expanded = (~padding_mask).float().unsqueeze(-1)
        masked_mean = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Regression
        output = self.regressor(masked_mean)
        return {self.cost_target_name: output}

def log_scale_mapping(x, max_value=2000.0):
    """Map values between 0 and max_value to log space"""
    max_value_tensor = torch.tensor(max_value, device=x.device)
    return torch.log1p(x) / torch.log1p(max_value_tensor)

def inverse_log_scale_mapping(x, max_value=2000.0):
    """Map values from log space back to original scale"""
    max_value_tensor = torch.tensor(max_value, device=x.device)
    return torch.expm1(x * torch.log1p(max_value_tensor))

class RegressionDataset(Dataset):
    def __init__(self, features, targets, target_range=None):
        self.features = features
        # Use target_range if provided, otherwise default to [0, 2000]
        self.target_range = target_range or [0, 2000]
        self.max_value = self.target_range[1]
        self.targets = log_scale_mapping(targets, self.max_value)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_model(model, train_loader, val_loader, config=None):
    # Set default values
    epochs = 100
    lr = 1e-4
    weight_decay = 0.01
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Override with config values if provided
    if config:
        epochs = config.get('epochs', epochs)
        lr = config.get('learning_rate', lr)
        weight_decay = config.get('weight_decay', weight_decay)
        if config.get('device'):
            device_name = config.get('device')
    
    device = torch.device(device_name)
    model = model.to(device)
    
    # Setup optimizer with config parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Use config for scheduler if provided
    scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs[model.cost_target_name], batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                val_loss += criterion(outputs[model.cost_target_name], batch_targets).item()
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Use config save path if provided
            save_path = 'best_gcan_model.pth'
            if config and config.get('model_checkpoint_path'):
                save_path = config.get('model_checkpoint_path')
            torch.save(model.state_dict(), save_path)

def predict(model, features, config=None):
    # Set default values
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_value = 2000.0
    
    # Override with config values if provided
    if config:
        if config.get('device'):
            device_name = config.get('device')
        if config.get('target_range'):
            max_value = config.get('target_range')[1]
    
    device = torch.device(device_name)
    model.eval()
    
    with torch.no_grad():
        features = features.to(device)
        outputs = model(features)
        # Convert back from log scale
        return inverse_log_scale_mapping(outputs[model.cost_target_name], max_value)

# Usage example:
if __name__ == "__main__":
    # This section is for testing only and should use config in real applications
    from omegaconf import OmegaConf
    
    # Load config
    try:
        config = OmegaConf.load('configs/models/gcan_v1.0.0.yaml')
        print("Using configuration from file")
    except:
        print("Using default configuration")
        config = {
            'model_kwargs': {
                'input_shape': [20, 768],
                'num_layers': 6,
                'num_heads': 8,
                'hidden_dim': 2048,
                'dropout': 0.1,
                'target_name': 'cost_target'
            },
            'trainer_kwargs': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'dataset_kwargs': {
                'target_range': [0, 2000]
            }
        }
    
    # Create model using config parameters
    model_kwargs = config.get('model_kwargs', {})
    model = GCAN(**model_kwargs)
    
    # Example data (replace with your actual data)
    batch_size = config.get('trainer_kwargs', {}).get('batch_size', 32)
    input_shape = model_kwargs.get('input_shape', [20, 768])
    target_range = config.get('dataset_kwargs', {}).get('target_range', [0, 2000])
    
    # Create synthetic data for testing
    train_features = torch.randn(1000, input_shape[0], input_shape[1])
    train_targets = torch.rand(1000) * target_range[1]
    val_features = torch.randn(200, input_shape[0], input_shape[1])
    val_targets = torch.rand(200) * target_range[1]
    
    # Create data loaders
    train_dataset = RegressionDataset(train_features, train_targets, target_range)
    val_dataset = RegressionDataset(val_features, val_targets, target_range)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train model with config
    train_model(model, train_loader, val_loader, config.get('trainer_kwargs', {}))
    
    # Make predictions
    test_features = torch.randn(10, input_shape[0], input_shape[1])
    predictions = predict(model, test_features, config.get('dataset_kwargs', {}))
    print("Predictions:", predictions)