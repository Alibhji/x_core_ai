MODEL_REGISTRY = {}

def register_model(name):
    """Decorator to register a model."""
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name} is already registered!")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name,*args, **kwargs):
    """Retrieve a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} is not registered.")
    return MODEL_REGISTRY[name](*args, **kwargs)