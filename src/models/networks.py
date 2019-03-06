import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ShallowEEGNet(nn.Module):
    """Convolution neural network class for eeg classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT):
        super(ShallowEEGNet, self).__init__()
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(1, 15), stride=1, bias=False),
            nn.Conv2d(20, 20, kernel_size=(10, 10), stride=1, bias=False),
            nn.BatchNorm2d(20, momentum=0.1, affine=True)
            )

        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))

        self.net_2 = nn.Sequential(
            nn.Conv2d(20, OUTPUT, kernel_size=(11, 11), stride=1),
            nn.LogSoftmax()
            )

    def forward(self, x):
        x = x.view(-1, 1, n_electrodes, epoch_length*s_freq)
        out = self.net_1(x)
        out = out*out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.net_2(out)
        out = torch.squeeze(out)

        return out
