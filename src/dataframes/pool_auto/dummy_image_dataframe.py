from ..registry import register_dataframe
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Union
import random
import string

@register_dataframe("dummy_image_dataframe")
class DummyImageDataframe:
    def __init__(self , 
                 image_size=224,
                 sequence_length=40,
                 number_of_samples=1000,
                 max_title_length=100,
                 max_description_length=1000,
                 num_tags=10,
                 num_categories=10,
                 train_portion=0.8,
                 val_portion=0.1,
                 test_portion=0.1):
        
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.number_of_samples = number_of_samples
        self.max_title_length = max_title_length
        self.max_description_length = max_description_length
        self.num_tags = num_tags
        self.num_categories = num_categories



        self.df_train , self.df_val , self.df_test = self.generate_dummy_data(train_portion, val_portion, test_portion)

    def tokenizer(self, text):
        return self.tokenizer(text, return_tensors='pt')


    def title_generator(self, length=100):
        random_title = "this is a test title for a random sequence of images!!!"
        random_title = random_title[:100] 
        return random_title

    def generate_dummy_data(self , train_portion=0.8 , val_portion=0.1 , test_portion=0.1):
            titles = []
            # Generate random image data
            images = [np.random.randn(self.sequence_length, 3, self.image_size, self.image_size) for _ in range(self.number_of_samples)]
            for i in range(self.number_of_samples):
                tokenized_title = self.title_generator(self.max_title_length)
                titles.append(tokenized_title)


            df = pd.DataFrame({
                'images': list(images),
                'title': list(titles)
            })   

            # Split the dataframe into train, val, and test sets
            train_df = df.sample(frac=train_portion)
            val_df = df.sample(frac=val_portion)
            if test_portion > 0:
                test_df = df.sample(frac=test_portion)
            else:
                test_df = pd.DataFrame()
            
            return train_df, val_df, test_df

    