from transformers import AutoTokenizer

class TokenizerZoo:
    # Class variable to store tokenizer instances
    _tokenizers = {
        'bert-base-uncased': AutoTokenizer.from_pretrained('bert-base-uncased'),
        # 'bert-large-uncased': AutoTokenizer.from_pretrained('bert-large-uncased'),
        # 'bert-base-cased': AutoTokenizer.from_pretrained('bert-base-cased'),
        # 'bert-large-cased': AutoTokenizer.from_pretrained('bert-large-cased'),
        # 'bert-base-multilingual-cased': AutoTokenizer.from_pretrained('bert-base-multilingual-cased'),
        # 'bert-large-multilingual-cased': AutoTokenizer.from_pretrained('bert-large-multilingual-cased'),
        
    }
    
    @classmethod
    def get_tokenizer(cls, tokenizer_name):
        """
        Get or create a tokenizer instance from the class dictionary.
        
        Args:
            tokenizer_name (str): Name of the pretrained tokenizer
            
        Returns:
            AutoTokenizer: The requested tokenizer instance
        """
        if tokenizer_name not in cls._tokenizers:
            cls._tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        return cls._tokenizers[tokenizer_name]
    
    def __init__(self, tokenizer_name):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = self.get_tokenizer(tokenizer_name)

    def get_instance_tokenizer(self):
        """
        Get this instance's tokenizer
        
        Returns:
            AutoTokenizer: The tokenizer instance
        """
        return self.tokenizer

