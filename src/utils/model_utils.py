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



