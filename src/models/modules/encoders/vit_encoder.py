from transformers import ViTModel
import torch.nn as nn
import torch

class ViTEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, images):  # (B, S, 3, 224, 224)
        B, S, C, H, W = images.shape
        images = images.view(B * S, C, H, W)  # Flatten sequence
        vit_out = self.vit(images).last_hidden_state[:, 0, :]  # CLS token
        vit_out = self.proj(vit_out)
        return vit_out.view(B, S, -1)  # Restore to (B, S, hidden_size)