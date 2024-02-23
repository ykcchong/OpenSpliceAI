import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import sys
import time

# class ResidualUnit(nn.Module):
#     def __init__(self, l, w, ar):
#         super(ResidualUnit, self).__init__()
#         self.bn1 = nn.BatchNorm1d(l)
#         self.conv1 = nn.Conv1d(l, l, w, padding=ar, dilation=ar)
#         self.bn2 = nn.BatchNorm1d(l)
#         self.conv2 = nn.Conv1d(l, l, w, padding=ar, dilation=ar)

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(x))
#         out = self.conv1(out)
#         out = F.relu(self.bn2(out))
#         out = self.conv2(out)
#         out += residual
#         return out
    


# class ResidualUnit(nn.Module):
#     def __init__(self, l, w, ar, bot_mul=1):
#         super().__init__()
#         bot_channels = int(round(l * bot_mul))
#         self.batchnorm1 = nn.BatchNorm1d(l)
#         self.relu = nn.LeakyReLU(0.1)
#         self.batchnorm2 = nn.BatchNorm1d(l)
#         # self.C = bot_channels//CARDINALITY_ITEM
#         self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)#, groups=self.C)
#         self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)#, groups=self.C)

#     def forward(self, x):
#         # x1 = self.relu(self.batchnorm1(self.conv1(x)))
#         # x2 = self.relu(self.batchnorm2(self.conv2(x1)))
#         x1 = self.conv1(self.relu(self.batchnorm1(x)))
#         x2 = self.conv1(self.relu(self.batchnorm1(x1)))
#         return x + x2


# class Cropping1D(nn.Module):
#     def __init__(self, cropping):
#         super().__init__()
#         self.cropping = cropping
    
#     def forward(self, x):
#         return x[:, :, self.cropping[0]:-self.cropping[1]]

# class SpliceAI(nn.Module):
#     def __init__(self, L, W, AR):
#         super(SpliceAI, self).__init__()
#         self.initial_conv = nn.Conv1d(4, L, 1)
#         self.skip_conv = nn.Conv1d(L, L, 1)
#         self.residual_units = nn.ModuleList([ResidualUnit(L, W[i], AR[i]) for i in range(len(W))])
#         self.final_conv = nn.Conv1d(L, 3, 1)
#         self.CL = 2 * np.sum(AR*(W-1))
#         self.crop = Cropping1D((self.CL//2, self.CL//2))  # Adjust this based on your specific needs

#     def forward(self, x):
#         x = self.initial_conv(x)
#         skip = self.skip_conv(x)
#         for ru in self.residual_units:
#             x = ru(x)
#             skip += self.skip_conv(x)
#         # print(f"skip shape before: {skip.shape}")
#         # skip = F.pad(skip, (self.CL//2, self.CL//2), 'constant', 0)
#         skip = self.crop(skip)  # Apply cropping here
#         # print(f"skip shape after: {skip.shape}")
#         out = self.final_conv(skip)
#         return F.softmax(out, dim=1)
    


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

class SpliceAI(nn.Module):
    def __init__(self, L, W, AR):
        super(SpliceAI, self).__init__()
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.residual_units = nn.ModuleList([ResidualUnit(L, W[i], AR[i]) for i in range(len(W))])
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))  # Adjust this based on your specific needs

    def forward(self, x):
        x = self.initial_conv(x)
        for ru in self.residual_units:
            x = ru(x)
        x = self.crop(x)  # Apply cropping here
        out = self.final_conv(x)
        return F.softmax(out, dim=1)  # Consider returning logits during training
