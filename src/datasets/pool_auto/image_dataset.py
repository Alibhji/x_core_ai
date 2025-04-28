from ..registry import register_dataset
from torch.utils.data import Dataset
from x_core_ai.src.models.tokenizer_zoo import TokenizerZoo
       
@register_dataset("image_dataset")
class ImageDataset(Dataset):
    def __init__(self, 
                 df=None,
                 train=True,
                 image_size=224,
                 sequence_length=40,
                 storage_path=None,
                 tokenizer_name=None):
        self.image_size = image_size
        self.image_sequence_length = sequence_length
        self.storage_path = storage_path
        self.tokenizer = TokenizerZoo.get_tokenizer(tokenizer_name)

        if df is not None:
            self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def title_tokenizer(self, title, length=100):
        # Tokenize the title to get input_ids with [CLS] and [SEP]
        tokenized = self.tokenizer(title)
        input_ids = tokenized['input_ids']  # e.g., [101, ..., 102]

        #TODO id it correct to drop SEP fron decoder_input_ids? and drop cls from decoder_target_ids?
        # Decoder input IDs (input to model): exclude [SEP]
        decoder_input_ids = input_ids[:-1]  # [CLS, ..., last_token]
        # Decoder target IDs (what to predict): exclude [CLS]
        decoder_target_ids = input_ids[1:]  # [..., SEP]

        # Pad both to 'length' tokens
        pad_id = self.tokenizer.pad_token_id
        decoder_input_ids = decoder_input_ids[:length] + [pad_id] * max(0, length - len(decoder_input_ids))
        decoder_target_ids = decoder_target_ids[:length] + [pad_id] * max(0, length - len(decoder_target_ids))

        return {
            'decoder_input_ids': decoder_input_ids,   # input to model
            'decoder_target_ids': decoder_target_ids  # for loss
        }
            #TODO
            # Do you want the final tgt_title to also be max 100 characters worth of decoded tokens? (that's trickier and requires decoding back + measuring string lengths), or
            # Are you happy with fixed token output from a 100-character input (what this gives you)?   
        
    def get_sample(self ,idx):

        title_dict = self.title_tokenizer(self.df.iloc[idx]['title'])
     
        return {
            'inputs': {
                'images': self.df.iloc[idx]['images'],
                'tgt_title': title_dict['decoder_input_ids']
            },
            'targets': {
                'title': title_dict['decoder_target_ids']
            }
        }
      

    def __getitem__(self ,idx):
        sample = self.get_sample(idx)
        return sample
    
    
    
    










