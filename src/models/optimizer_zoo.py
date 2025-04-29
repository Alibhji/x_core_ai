import torch

class OptimizerZoo:
    """
    A class for managing and selecting optimizers based on the optimizer name.
    """
    # class variable to store optimizer classes
    _optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
    }

    @classmethod
    def get_optimizer(cls, optimizer_name):
        """
        Get optimizer class from the class dictionary.
        """
        if optimizer_name not in cls._optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return cls._optimizers[optimizer_name]
    
    def __init__(self, optimizer_name, model_parameters, **kwargs):
        self.optimizer_name = optimizer_name
        optimizer_class = self.get_optimizer(optimizer_name)
        self.optimizer = optimizer_class(model_parameters, **kwargs)

    def get_instance_optimizer(self):
        """
        Get this instance's optimizer
        """
        return self.optimizer

