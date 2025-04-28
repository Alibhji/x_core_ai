from ..modules.encoders.vit_encoder import ViTEncoder
from ..modules.heads.title_gen import TitleGenerator
from ..registry import register_model
import torch.nn as nn
import torch

@register_model("vit_encoder_decoder")
class ViTEncoderDecoder(nn.Module):
    def __init__(self, vocab_size=30522, 
                 hidden_size=768, 
                 freeze_vit_encoder=True):
        
        super().__init__()
        self.vit_encoder = ViTEncoder(hidden_size)
        self.title_generator = TitleGenerator(vocab_size, hidden_size)
        if freeze_vit_encoder:
            self.freeze_vit_encoder()


    def freeze_vit_encoder(self):
        for param in self.vit_encoder.parameters():
            param.requires_grad = False


    def forward(self, images, tgt_title):
        encoder_features = self.vit_encoder(images)
        title_logits = self.title_generator(encoder_features, tgt_title)
        return {'title': title_logits}
    
    @torch.no_grad()
    def generate(self, images, max_len=100, start_token_id=101):
        """
        Autoregressive title generation at inference time.

        Args:
            images (torch.Tensor): Batch of images [batch_size, 3, 224, 224]
            max_len (int): Maximum length of generated title
            start_token_id (int): ID of the start token ([CLS] or [SOS])

        Returns:
            dict with:
                'title': generated token IDs [batch_size, max_len]
                'logits': predicted logits [batch_size, max_len, vocab_size]
        """
        encoder_features = self.vit_encoder(images)
        
        batch_size = images.size(0)
        device = images.device
        
        # Initialize generated tokens
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        # Store logits at each time step
        logits_list = []

        for _ in range(max_len - 1):
            # Predict next token
            logits = self.title_generator(encoder_features, generated)  # (batch_size, seq_len, vocab_size)
            
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size) — logits for last token
            
            logits_list.append(next_token_logits.unsqueeze(1))  # keep dimension (batch_size, 1, vocab_size)
            
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
            
            # Append predicted token
            generated = torch.cat([generated, next_token], dim=1)
        
        # After loop, stack all collected logits
        logits = torch.cat(logits_list, dim=1)  # (batch_size, max_len-1, vocab_size)
        
        # Add dummy logits for the first start token
        start_logits = torch.zeros((batch_size, 1, logits.size(-1)), device=device)
        logits = torch.cat([start_logits, logits], dim=1)  # (batch_size, max_len, vocab_size)
        
        return {
            'title': generated,      # (batch_size, max_len) token IDs
            'title_logits': logits          # (batch_size, max_len, vocab_size) raw logits
        }
