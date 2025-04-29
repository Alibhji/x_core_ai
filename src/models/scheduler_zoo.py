import torch

class SchedulerZoo:
    """
    A class for managing and selecting schedulers based on the scheduler name.
    """
    # class variable to store scheduler classes
    _schedulers = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'cosine_with_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }

    @classmethod
    def get_scheduler(cls, scheduler_name):
        """
        Get scheduler class from the class dictionary.
        """
        if scheduler_name not in cls._schedulers:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        return cls._schedulers[scheduler_name]
    
    def __init__(self, scheduler_name, optimizer, **kwargs):
        self.scheduler_name = scheduler_name
        scheduler_class = self.get_scheduler(scheduler_name)
        
        # Filter kwargs based on scheduler type
        if scheduler_name == 'cosine':
            valid_kwargs = {
                'T_max': kwargs.get('T_max', 100),
                'eta_min': kwargs.get('eta_min', 0),
                'last_epoch': kwargs.get('last_epoch', -1)
            }
        elif scheduler_name == 'plateau':
            valid_kwargs = {
                'mode': kwargs.get('mode', 'min'),
                'factor': kwargs.get('factor', 0.1),
                'patience': kwargs.get('patience', 10),
                'verbose': kwargs.get('verbose', True),
                'threshold': kwargs.get('threshold', 1e-4),
                'threshold_mode': kwargs.get('threshold_mode', 'rel'),
                'cooldown': kwargs.get('cooldown', 0),
                'min_lr': kwargs.get('min_lr', 0),
                'eps': kwargs.get('eps', 1e-8)
            }
        elif scheduler_name == 'cosine_with_restarts':
            valid_kwargs = {
                'T_0': kwargs.get('T_0', 50),
                'T_mult': kwargs.get('T_mult', 1),
                'eta_min': kwargs.get('eta_min', 0),
                'last_epoch': kwargs.get('last_epoch', -1)
            }
        
        self.scheduler = scheduler_class(optimizer, **valid_kwargs)

    def get_instance_scheduler(self):
        """
        Get this instance's scheduler
        """
        return self.scheduler
