import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)
def load_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    logger.info(f' >>> [Single Training] --> Load weights from: {weights_path}')

def save_weights(model, weights_path , epoch  , distributed = False ):
    if distributed:
        global_rank =  dist.get_rank() if distributed else 0
        local_rank = dist.get_local_rank() if distributed else 0
        if global_rank == 0:
            torch.save(model.state_dict(), weights_path + f'{epoch}.pth')
            logger.info(f' >>> [DDP Training] --> Rank: {local_rank} --> Save weights at epoch: {epoch}')
            
    else:
        torch.save(model.state_dict(), weights_path + f'{epoch}.pth')
        logger.info(f' >>> [Single Training] --> Save weights at epoch: {epoch}')

def analyze_model_size(model):
    """
    Analyze model parameters, estimate size, and detect weight types.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with model statistics
    """
    if model is None:
        logger.warning("No model to analyze")
        return {}
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # Parameter type counters and memory usage
    param_types = {}
    total_memory = 0
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        
        # Count trainable vs non-trainable
        if param.requires_grad:
            trainable_params += param_size
        else:
            non_trainable_params += param_size
            
        # Count parameter types and memory
        dtype_name = str(param.dtype).split('.')[-1]
        if dtype_name not in param_types:
            # Get bytes per element for this dtype
            bytes_per_element = torch.empty(1, dtype=param.dtype).element_size()
            param_types[dtype_name] = {
                'count': 0,
                'bytes_per_element': bytes_per_element,
                'memory_mb': 0
            }
        
        param_types[dtype_name]['count'] += param_size
        memory = param_size * param_types[dtype_name]['bytes_per_element']
        param_types[dtype_name]['memory_mb'] += memory / (1024 * 1024)
        total_memory += memory
    
    # Convert memory to MB
    total_memory_mb = total_memory / (1024 * 1024)
    
    # Prepare results
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'param_types': param_types,
        'total_memory_mb': total_memory_mb,
        'model_name': model.__class__.__name__
    }
    
    # Print results
    model_name = results['model_name']
    delimiter = "=" * (len(model_name) + 16)
    print(f"\n{delimiter}")
    print(f"== Model: {model_name} ==")
    print(f"{delimiter}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Model size: {total_memory_mb:.2f} MB")
    
    print("\nParameter types:")
    for dtype, info in param_types.items():
        percent = info['count'] / total_params * 100
        memory_percent = info['memory_mb'] / total_memory_mb * 100
        print(f"  {dtype}: {info['count']:,} parameters ({percent:.2f}%), {info['memory_mb']:.2f} MB ({memory_percent:.2f}%)")
    
    print(f"{delimiter}\n")
    
    return results



