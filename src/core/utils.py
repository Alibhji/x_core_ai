import torch
import numpy as np
import sklearn.metrics as metrics

def custom_collate_fn(batch):
        collated_batch = {}

        # Get all keys in the dataset
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]

            if isinstance(values[0], np.ndarray):
                try:
                    collated_batch[key]= torch.as_tensor(np.stack(values))
                except:
                    print('issue in photofeatures')
            elif isinstance(values[0], list):
                collated_batch[key] = torch.stack([torch.tensor(v) for v in values])

            elif isinstance(values[0], int) or isinstance(values[0], float):  # Handle numerical values
                collated_batch[key] = torch.tensor(values)
                
            elif isinstance(values[0], torch.Tensor) or isinstance(values[0], np.ndarray):
                values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in values]
                collated_batch[key] = torch.stack(values)  # Stack tensors

            elif isinstance(values[0], str) or isinstance(values[0], list):  # Handle strings/lists
                collated_batch[key] = values  # Keep them as lists
                
            elif isinstance(values[0], dict):
                collated_batch[key] = custom_collate_fn(values)
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(values[0])}")

        return collated_batch

