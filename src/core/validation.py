from .core_base import Core
import torch
from .metrics import Metrics

class Validation(Core):
    """Validation class for model validation"""

    def __init__(self, config, 
                 package_name='x_core_ai.src',
                 df_val=None,
                 val_dataloader=None):
        super().__init__(config, package_name)
        self.setup_model()
        if df_val is None and val_dataloader is None:
            df = self._get_dataframe(self.config['data_name'], **self.config['data_kwargs'])
            dataset = self.create_dataset(df.df_val, train=False)
            self.val_dataloader = self.create_dataloader(dataset, train=False)
        elif df_val is not None:
            dataset = self.create_dataset(df_val, train=False)
            self.val_dataloader = self.create_dataloader(dataset, train=False)
        elif val_dataloader is not None:
            self.val_dataloader = val_dataloader
        print("length of val_dataloader: ", len(self.val_dataloader))

    def setup_model(self):
        """Setup model for inference"""
        self.model_generator()
        self.model_to_device()
        self.model.eval()
        print("Model setup complete")

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

    # @torch.no_grad()
    def one_eopch_validation(self, convert_to_string=False):
        """One epoch of validation"""
        self.metrics = Metrics(metrics_kwargs=self.config.get('metrics_kwargs', {}))
        all_results = {}
        
        for i,batch in enumerate(self.val_dataloader):
            print(f"Batch {i+1} of {len(self.val_dataloader)}")
            inputs, targets = self.get_inputs_targets(batch, drop_keys=['tgt_title'])
            # Get model outputs - could be a dictionary with both tokens and logits
            outputs = self.model.generate(**inputs)
            # Calculate metrics using appropriate output types
            batch_results = self.metrics.calculate_metrics(targets, outputs)
            for key, value in batch_results.items():
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(value)
            
            print(self.metrics.get_metric_string(batch_metrics=True))
            
            # Convert token IDs to strings if needed
            if convert_to_string:
                for task in outputs:
                    if task in ['title'] and isinstance(outputs[task], torch.Tensor):
                        outputs[f"{task}_text"] = self.tokenizer.decode(outputs[task], skip_special_tokens=True)
            
        print(self.metrics.get_metric_string(batch_metrics=False))
        return all_results
        

    def get_validation_metrics(self):
        """
        Run validation and return metrics
        
        Returns:
            Dictionary of metrics with keys like 'title_BLEU', 'title_accuracy'
        """
        with torch.no_grad():
            results = self.one_eopch_validation()
        
        # Calculate mean for each metric
        final_metrics = {}
        for key, values in results.items():
            if values:  # Check if values exist
                final_metrics[key] = torch.tensor(values).mean().item()
        
        return final_metrics
