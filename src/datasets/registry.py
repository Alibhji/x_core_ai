# datasets/registry.py
DATASET_REGISTRY = {}

def register_dataset(name):
    """Decorator to register a dataset."""
    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset {name} is already registered!")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name,*args, **kwargs):
    """Retrieve a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} is not registered.")
    return DATASET_REGISTRY[name](*args, **kwargs)