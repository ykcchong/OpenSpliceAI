import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import h5py
import sys
import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a positional encoding matrix with shape [max_len, d_model].
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Make sure positional encoding is at the same device as x.
        pe = self.pe[:, :x.size(1)].to(x.device)
        return x + pe


class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)

    def forward(self, x):
        residual = x
        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.conv2(self.relu(self.batchnorm1(out)))
        return residual + out

class Cropping1D(nn.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]

# class SpliceAI(nn.Module):
#     def __init__(self, L, W, AR):
#         super(SpliceAI, self).__init__()
#         self.initial_conv = nn.Conv1d(4, L, 1)
#         self.residual_units = nn.ModuleList([ResidualUnit(L, W[i], AR[i]) for i in range(len(W))])

#         self.num_heads = 4
#         # Add MultiheadAttention here; ensure embedding dim is divisible by num_heads
#         self.attention = nn.MultiheadAttention(embed_dim=L, num_heads=self.num_heads, batch_first=True)
#         self.dropout = nn.Dropout(0.1)  # Dropout after attention
#         self.layer_norm1 = nn.LayerNorm(L)  # Normalize before attention
#         self.layer_norm2 = nn.LayerNorm(L)  # Normalize before final conv

#         self.positional_encoding = PositionalEncoding(L)  # Implement or use a predefined function

#         self.final_conv = nn.Conv1d(L, 3, 1)
#         self.CL = 2 * np.sum(AR * (W - 1))
#         self.crop = Cropping1D((self.CL//2, self.CL//2))  # Adjust this based on your specific needs

#     def forward(self, x):
#         x = self.initial_conv(x)
#         for ru in self.residual_units:
#             x = ru(x)
#         x = self.crop(x)  # Apply cropping here

#         # Permute for attention layer, apply attention, then permute back
#         x = x.permute(0, 2, 1)  # Change shape to (N, L, C) for MultiheadAttention
#         x, _ = self.attention(x, x, x)
#         x = x.permute(0, 2, 1)  # Permute back to (N, C, L)

#         out = self.final_conv(x)
#         return F.softmax(out, dim=1)  # Consider returning logits during training


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR):
        super(SpliceAI, self).__init__()
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.residual_units = nn.ModuleList([ResidualUnit(L, W[i], AR[i]) for i in range(len(W))])

        self.num_heads = 4  # Ensure embedding dimension (L) is divisible by num_heads
        self.attention = nn.MultiheadAttention(embed_dim=L, num_heads=self.num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm([L])  # Adjust dimension if necessary
        self.layer_norm2 = nn.LayerNorm([L])  # Adjust dimension if necessary

        self.positional_encoding = PositionalEncoding(L)
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))

    def forward(self, x):
        x = self.initial_conv(x)
        for ru in self.residual_units:
            x = ru(x)
        x = self.crop(x)

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, channels) for attention
        
        # Apply positional encoding correctly.
        pos_encoded = self.positional_encoding(x)
        x = x + pos_encoded  # Add positional encoding
        x = self.layer_norm1(x)  # Apply normalization before attention

        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = self.layer_norm2(x)  # Apply normalization after attention
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, channels, seq_len)

        out = self.final_conv(x)
        return F.softmax(out, dim=1)  # Consider returning logits during training for numerical stability
