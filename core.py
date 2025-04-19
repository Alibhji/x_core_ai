'''@author: ali.bhji'''
import os
import importlib
import pandas as pd
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
# from src.utils.utils import load_model_weights
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


class Core:
    def __init__(self ,config , ):
        self.config = config
        self.inititiaze_package()

        gpus = config.get('gpus' , list(range(torch.cuda.device_count())))
        primary_gpu = gpus[0] if gpus else 0
        self.device = torch.device(f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu")
        if self.config['distributed']:
            self.device = f"cuda:{primary_gpu}"
        # Import all models from the 'pool' directory
        

    def inititiaze_package(self , package_name ='x_core_ai.src'):
        package_name = package_name
        spec = importlib.util.find_spec(package_name)
        self.root_package_path = os.path.dirname(os.path.abspath(spec.origin))
        dirs = ['dataset', 'model', 'dataframe']
        for dir in dirs:
            if self.config.get(dir+'_name'):
                self.package_importer(dir+'s'+os.sep+'pool_auto' , self.config[dir+'_name'])

     
    def model_generator(self):
        self.model = self.get_model(self.config["model_name"], 
                                self.config["model_kwargs"])
        load_model_weights(self.model , self.config.get('weights_path'))
        
    def model_to_device(self):
        
        gpus = self.config['gpus']
        if self.config["distributed"]:
            print(f' >>> [DDP Training] --> Rank: {self.config["dist_kwargs"]["local_rank"]} ')
            self.device = torch.device(f"cuda:{self.config['dist_kwargs']['local_rank']}" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            find_unused_parameters = self.config.get('find_unused_parameters' , False)
            self.model = DDP(self.model, 
                            device_ids=[self.config['dist_kwargs']['local_rank']],
                            find_unused_parameters = find_unused_parameters,
                            output_device= self.config['dist_kwargs']['local_rank'] #TODO check this line
                            )
            
        elif len(gpus)>1:
            print(f' >>> [DP Training] --> with {len(gpus)} GPUs')
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model )
        else:
            self.model.to(self.device)
            print(f' >>> [Single GPU Training]')

    def fix_batch_in_correct_format_if_needed(self, batch):
        try:
            primary_key = batch.pop('primary_key') 
        except:
            batch = self.custom_collate_fn(batch)
            primary_key = batch.pop('primary_key')
        return primary_key, batch

    def package_importer(self,relative_path_inside_package , module_list):
        module_list = module_list if isinstance(module_list , list) else [module_list]
        relative_module_path = os.path.join(*self.root_package_path.split(os.sep)[-2:], *relative_path_inside_package.split('/')).replace(os.sep, '.')
        for file in module_list:
            module_name = f"{relative_module_path}.{file}"
            print(f"Importing module: {module_name}")
            importlib.import_module(module_name)

    def _get_dataframe(self,name, *vars, **kwargs):
        return src.dataframes.registry.get_dataframe(name, *vars, **kwargs)

    def get_dataset(self,name, *vars, **kwargs):
        return src.datasets.registry.get_dataset(name, *vars, **kwargs)

    def get_model(self,name, *vars, **kwargs):
        return src.models.registry.get_model(name, *vars, **kwargs)
    
    def get_dataframes(self):
        if isinstance(self.config['data_name'] , list):
            dataframes = []
            for i,name in enumerate(self.config['data_name']):
                kwargs_name = 'dataframe_kwargs'
                str_ = f'_{i}' if i!=0 else ''
                kwargs_name += str_
                dataframes.append( self._get_dataframe(name, **self.config[kwargs_name] ))
                df_train = pd.concat([df_.df_train for df_ in dataframes])
                df_test = pd.concat([df_.df_test for df_ in dataframes])
        else:
            dataframes = self._get_dataframe(self.config['data_name'], **self.config['dataframe_kwargs'] )
            df_train = dataframes.df_train
            df_test  = dataframes.df_test
        return df_train , df_test

    def dataset_generator(self , df , isTrain):
        return self.get_dataset(self.config["dataset_name"], 
                                            df = df,  
                                            train = isTrain,
                                        **self.config["dataset_kwargs"])

    def get_dataloader(self , dataset ,isTrain ,prefetch_factor = 4):
        sampler = None
        if self.config["distributed"]:
            sampler = DistributedSampler(dataset, num_replicas=self.config["dist_kwargs"]["world_size"], rank=self.config["dist_kwargs"]["local_rank"], shuffle=True)
        else:
            # for DP Training total batch of data should be multiply by number of gpus
            if isTrain:
                new_batch = self.config['dataloader_kwargs_train']['batch_size'] * len(self.config['gpus'])
                self.config['dataloader_kwargs_train']['batch_size'] = new_batch
            else:
                new_batch =  self.config['dataloader_kwargs_val']['batch_size'] * len(self.config['gpus'])
                self.config['dataloader_kwargs_val']['batch_size'] = new_batch

            print(f">>> total batch_size changed from {self.config['batch_size']} to {new_batch}")   

        if isTrain:
            return DataLoader(dataset, 
                            shuffle=not self.config["distributed"],
                            sampler=sampler,
                            prefetch_factor= prefetch_factor,
                            **self.config['dataloader_kwargs_train'])
        else:                                    
            return DataLoader(dataset, shuffle=False , 
                                    prefetch_factor= prefetch_factor,
                                    **self.config['dataloader_kwargs_val'],
                                    collate_fn= self.custom_collate_fn)

    def get_train_test_dataloaders(self):
        if 'mongo' in self.config.get('dataset_name',''):
            train_claim_set_names = self.config['train_claim_set_names']
            val_claim_set_names   = self.config['val_claim_set_names']
            parquet_claim_ids_path_train = self.config.get('parquet_claim_ids_path_train')
            parquet_claim_ids_path_val = self.config.get('parquet_claim_ids_path_val')

            return (self.get_mongodb_dataloaders(isTrain = True ,
                                                claim_set_names = train_claim_set_names,
                                                parquet_claim_ids_path = parquet_claim_ids_path_train) 
                                                , 
                    self.get_mongodb_dataloaders(isTrain = False ,
                                                claim_set_names = val_claim_set_names,
                                                parquet_claim_ids_path = parquet_claim_ids_path_val))
        else:
            df_train, df_test = self.get_dataframes()
            dataset_train = self.dataset_generator(df_train, isTrain = True)
            dataset_val  = self.dataset_generator(df_test, isTrain = False)
            return self.get_dataloader(dataset_train, isTrain = True) , self.get_dataloader(dataset_val, isTrain = False)

    def get_mongodb_dataloaders(self , isTrain , claim_set_names, parquet_claim_ids_path):
        rank = self.config['dist_kwargs'].get('local_rank' , 0)
        world_size = self.config['dist_kwargs'].get('world_size', 1)
        mongodb_dataloder =  self.get_dataset(self.config["dataset_name"],
train = isTrain,
                                        claim_set_names = claim_set_names,
                                        rank= rank , 
                                        world_size = world_size,
                                        # parquet_claim_ids_path = parquet_claim_ids_path,
                                        **self.config["dataset_kwargs"])
        return mongodb_dataloder.get_dataloader()
    
    @staticmethod
    def cleanup():
        """Destroy the distributed process group."""
        dist.destroy_process_group()
    
    @staticmethod
    def gather_tensors(tensor, world_size):
        """
        Gather tensors from all GPUs using all_gather.
        Returns a concatenated tensor across all GPUs.
        """
        tensor = tensor.detach()  # Detach to prevent unnecessary gradient computation
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)  # Collect from all GPUs
        return torch.cat(gathered_tensors, dim=0)  # Concatenate all predictions
    
    @staticmethod
    def gather_on_rank0(tensor_dict, rank, world_size):
        """
        Gather dictionary of tensors from all GPUs on rank 0.
        Only rank 0 will have the complete tensor dictionary.
        
        Args:
            tensor_dict: Dictionary of tensors to gather
            rank: Current process rank
            world_size: Total number of processes
        Returns:
            Dictionary of concatenated tensors on rank 0, None on other ranks
        """
        # Detach all tensors in the dictionary
        tensor_dict = {k: v.detach() for k, v in tensor_dict.items()}
        
        # Initialize gathered tensors dictionary on rank 0
        gathered_tensors = None
        if rank == 0:
            gathered_tensors = {
                k: [torch.zeros_like(v) for _ in range(world_size)]
                for k, v in tensor_dict.items()
            }
        
        # Gather each tensor in the dictionary
        for key, tensor in tensor_dict.items():
            if rank == 0:
                dist.gather(tensor, gathered_tensors[key], dst=0)
            else:
                dist.gather(tensor, dst=0)
        
        # Concatenate tensors on rank 0
        if rank == 0:
            return {
                k: torch.cat(tensors, dim=0)
                for k, tensors in gathered_tensors.items()
            }
        return None
    
    @staticmethod
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

            elif isinstance(values[0], int) or isinstance(values[0], float):  # Handle numerical values
                collated_batch[key] = torch.tensor(values)
                
            elif isinstance(values[0], torch.Tensor) or isinstance(values[0], np.ndarray):
                values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in values]
                collated_batch[key] = torch.stack(values)  # Stack tensors

            elif isinstance(values[0], str) or isinstance(values[0], list):  # Handle strings/lists
                collated_batch[key] = values  # Keep them as lists
                
            elif isinstance(values[0], dict):
                collated_batch[key] = Core.custom_collate_fn(values)
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(values[0])}")

        return collated_batch