# dataframes/registry.py
DATAFRAME_REGISTRY = {}

def register_dataframe(name):
    """Decorator to register a dataframe."""
    def decorator(cls):
        if name in DATAFRAME_REGISTRY:
            raise ValueError(f"dataframe {name} is already registered!")
        DATAFRAME_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataframe(name,*args, **kwargs):
    """Retrieve a dataframe by name."""
    if name not in DATAFRAME_REGISTRY:
        raise ValueError(f"dataframe {name} is not registered.")
    return DATAFRAME_REGISTRY[name](*args, **kwargs)