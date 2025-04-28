import torch
import torch.nn as nn

class TitleGenerator(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768 , seq_len=100):  # match ViT hidden size
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_size))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=2
        )

        self.head = nn.Linear(hidden_size, vocab_size)
        self.seq_len = seq_len

    def forward(self, encoder_features, tgt_ids):
        '''
        input:
            encoder_features: (B, 40, hidden_size)
            tgt: (B, max_title_length) this is the target title sequence without the start token for example if the max_title_length is 10, the tgt should be (B, 9)
        output:
            logits: (B, max_title_length, vocab_size)
        '''
        # for images: (B, S, 3, 224, 224)
        # encoder_features (B, S, hidden_size) , tgt_ids: (B, seq_len)
        tgt_emb = self.embedding(tgt_ids) + self.positional_encoding[:, :tgt_ids.size(1), :]
        out = self.decoder(tgt=tgt_emb, memory=encoder_features)
        logits = self.head(out)  # (B, seq_len, vocab_size)
        return logits
    
