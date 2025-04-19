from ..registry import register_dataframe
import torch
import pandas as pd
    

@register_dataframe("dummy_dataframe")
class DummyDataframe:
    def __init__(self, 
                 photo_feature_sequence_length: int,
                 photo_feature_dim: int,
                 target_name: str,
                 target_type: str,
                 target_range: list,
                 number_of_samples: int,
                 ):
        """ this dataframe is used for testing and demonstration purposes"""
        self.photo_feature_sequence_length = photo_feature_sequence_length
        self.photo_feature_dim = photo_feature_dim
        self.target_name = target_name
        self.target_type = target_type
        self.target_range = target_range
        self.number_of_samples = number_of_samples

        self.df_train = self.generate_dataframe()
        self.df_val = self.df_train.sample(frac=0.1)
        self.df_test = self.df_train.sample(frac=0.1)


    def generate_dataframe(self):
        # Generate random photo features
        photo_features = torch.randn(self.number_of_samples, self.photo_feature_sequence_length, self.photo_feature_dim)
        
        # Generate random target values
        if self.target_type == "regression":
            targets = torch.randint(self.target_range[0], self.target_range[1], (self.number_of_samples,))  
        elif self.target_type == "classification":
            targets = torch.randint(0, 2, (self.number_of_samples,))
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")    
        
        # Create a pandas DataFrame - convert tensors to lists to avoid pandas dimension error
        df = pd.DataFrame({
            'photo_features': list(photo_features),
            self.target_name: list(targets)
        })  
        
        return df
    

    def save_dataframe(self):
        self.df_train.to_parquet('dataframes/auto_pool/dummy_dataframe_train.parquet')
        self.df_val.to_parquet('dataframes/auto_pool/dummy_dataframe_val.parquet')
        self.df_test.to_parquet('dataframes/auto_pool/dummy_dataframe_test.parquet')




