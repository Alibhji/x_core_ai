import torch
import numpy as np
from .core_base import Core

class Forecast(Core):
    def __init__(self, config, package_name='x_core_ai.src'):
        """Forecast class for model inference and predictions"""
        super().__init__(config, package_name)
        self.prediction_mode = True
        self.setup_model()
        
    def setup_model(self):
        """Setup model for inference"""
        self.model_generator()
        self.model_to_device()
        self.model.eval()
        
    def predict(self, inputs, raw_output=False):
        """
        Make predictions with the model
        Args:
            inputs: Input data for the model
            raw_output: Whether to return raw model outputs
        Returns:
            Predictions from the model
        """
        # Handle GCAN model specifically which requires a tensor named photo_feat
        is_gcan = self.config.get('model_name') == 'gated_cross_attention'
        
        # Convert to appropriate format if needed
        if not isinstance(inputs, torch.Tensor):
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            elif isinstance(inputs, dict):
                # Handle dictionary inputs - extracting relevant tensor
                # For GCAN, use photo_feat directly
                if is_gcan and 'photo_feat' in inputs:
                    inputs = inputs['photo_feat']
                # Try standard keys in order of likelihood
                elif 'photo_feat' in inputs:
                    inputs = inputs['photo_feat']
                elif 'features' in inputs:
                    inputs = inputs['features']
                elif 'input' in inputs:
                    inputs = inputs['input']
                elif 'x' in inputs:
                    inputs = inputs['x']
                # If still a dict and not GCAN, leave as is for model to handle
                # If still a dict and is GCAN, we need to pick any tensor to use
                elif is_gcan:
                    # Find first tensor in dict to use as input
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs = v
                            break
            else:
                try:
                    inputs = torch.tensor(inputs, dtype=torch.float)
                except:
                    pass  # Leave as is if can't convert
                
        # Move to device if it's a tensor
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, dict):
            # Move tensor values to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # For GCAN model, ensure input is treated as photo_feat if it's a tensor
        if is_gcan and isinstance(inputs, torch.Tensor):
            with torch.no_grad():
                outputs = self.model(photo_feat=inputs)
        else:
            with torch.no_grad():
                outputs = self.model(inputs)
            
        if raw_output:
            return outputs
            
        # Process outputs if needed
        predictions = self.process_outputs(outputs)
        return predictions
    
    def process_outputs(self, outputs):
        """
        Process raw model outputs into usable predictions
        Override this method for custom output processing
        """
        # Default processing - depends on output format
        if isinstance(outputs, dict):
            # If model returns a dict, extract the main prediction
            target_name = self.config.get('target_name', 'cost_target')
            if target_name in outputs:
                predictions = outputs[target_name]
            else:
                # Take the first value as prediction
                predictions = next(iter(outputs.values()))
        elif isinstance(outputs, torch.Tensor):
            predictions = outputs
        else:
            predictions = outputs
            
        # Move to CPU for further processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            
        return predictions
    
    def batch_predict(self, dataloader=None, progress_callback=None):
        """
        Make predictions on a batch of data
        Args:
            dataloader: DataLoader containing batch data (uses default val dataloader if None)
            progress_callback: Optional callback for progress updates
        Returns:
            Batched predictions
        """
        # Get dataloader if not provided
        if dataloader is None:
            dataloader = self.get_val_dataloader()
            
        if dataloader is None:
            raise ValueError("No dataloader available for batch prediction")
            
        all_predictions = []
        all_inputs = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Handle batch format
                if isinstance(batch, dict):
                    # Keep original batch for input ID
                    batch_data = batch
                    if 'primary_key' in batch:
                        primary_key = batch['primary_key']
                    else:
                        primary_key = None
                else:
                    # Use fix_batch_format for tuple-style batches
                    primary_key, batch_data = self.fix_batch_format(batch)
                
                # Get predictions
                batch_predictions = self.predict(batch_data, raw_output=False)
                all_predictions.append(batch_predictions)
                
                # Store input identifiers if available
                if primary_key is not None:
                    all_inputs.append(primary_key)
                    
                # Update progress if callback provided
                if progress_callback and callable(progress_callback):
                    progress_callback(i, len(dataloader))
                    
        # Combine predictions
        if all(isinstance(p, np.ndarray) for p in all_predictions):
            final_predictions = np.vstack(all_predictions)
        else:
            final_predictions = all_predictions
            
        return final_predictions, all_inputs if all_inputs else None 