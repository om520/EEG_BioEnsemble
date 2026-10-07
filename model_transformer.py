import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class EEGTransformer(nn.Module):
    def __init__(self, in_channels=18, patch_size=16, seq_len=1024, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        self.patch_proj = nn.Linear(in_channels * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 18, 1024)
        B, C, T = x.shape
        # reshape to patches: (B, num_patches, C * patch_size)
        x = x.view(B, C, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, self.num_patches, C * self.patch_size)
        
        x = self.patch_proj(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches+1, d_model)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        cls_out = x[:, 0, :]
        out = self.fc(cls_out)
        return self.sigmoid(out).squeeze(-1)
