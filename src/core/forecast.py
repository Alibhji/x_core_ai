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
        print("Model setup complete")
        
    def predict(self, inputs):
        """
        Make predictions with the model
        Args:
            inputs: Input data for the model
            raw_output: Whether to return raw model outputs
        Returns:
            Predictions from the model
        """
        # convert all value of the dict to tensor
        inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        # forward pass use generate method
        outputs = self.model.generate(**inputs)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs


        
    def batch_predict(self, list_of_dicts=None, pytorch_dataloader=None, progress_callback=None):
        """
        Make predictions on a batch of data
        Args:
            list_of_dicts: List of dictionaries containing the input data
            pytorch_dataloader: Pytorch DataLoader containing the input data
            progress_callback: Optional callback for progress updates (receives current_batch, total_batches, batch_outputs)
        Returns:
            Batched predictions
        """
        predictions = []
        if list_of_dicts is not None:
            for batch in list_of_dicts:
                outputs = self.predict(batch)
                predictions.append(outputs)
        elif pytorch_dataloader is not None:
            for batch in pytorch_dataloader:
                outputs = self.predict(batch)   
                predictions.append(outputs)
        return predictions
    

    
