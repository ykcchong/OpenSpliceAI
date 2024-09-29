# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class ResidualUnit(nn.Module):
#     def __init__(self, channels, kernel_size, dilation):
#         super(ResidualUnit, self).__init__()
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=dilation)
#         self.bn2 = nn.BatchNorm1d(channels)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=dilation)

#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(x))
#         x = self.conv1(x)
#         x = F.relu(self.bn2(x))
#         x = self.conv2(x)
#         return x + residual

# class SpliceAI(nn.Module):
#     def __init__(self, L, W, AR):
#         super(SpliceAI, self).__init__()
#         assert len(W) == len(AR)
#         self.CL = 2 * np.sum(np.array(AR) * (np.array(W) - 1))
#         self.initial_conv = nn.Conv1d(4, L, 1)
#         self.skip_conv = nn.Conv1d(L, L, 1)
        
#         self.residual_units = nn.ModuleList([
#             ResidualUnit(L, W[i], AR[i]) for i in range(len(W))
#         ])
        
#         self.skip_connections = nn.ModuleList([
#             nn.Conv1d(L, L, 1) for _ in range(len(W) // 4 + 1)
#         ])
        
#         self.final_conv = nn.Conv1d(L, 3, 1)

#     def forward(self, x):
#         x = self.initial_conv(x)
#         skip = self.skip_conv(x)
        
#         for i, unit in enumerate(self.residual_units):
#             x = unit(x)
#             if (i + 1) % 4 == 0 or (i + 1) == len(self.residual_units):
#                 dense = self.skip_connections[i // 4](x)
#                 skip = skip + dense
        
#         skip = skip[:, :, self.CL//2:-self.CL//2]
#         output = self.final_conv(skip)
#         return F.softmax(output, dim=1)

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)

    def forward(self, x, y):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return x + out, y


class Cropping1D(nn.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]


class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR):
        super(SpliceAI, self).__init__()
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, w, r))
            if (i+1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))

    def forward(self, x):
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            x, skip = m(x, skip)
        final_x = self.crop(skip)
        out = self.final_conv(final_x)
        return F.softmax(out, dim=1)
