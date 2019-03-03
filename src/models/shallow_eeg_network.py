import torch
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from visdom import Visdom
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

def weights_init(model):
    """Xavier normal weight initialization for the given model. 

    Parameters
    ----------
    model : pytorch model for random weight initialization
    Returns
    -------
    pytorch model with xavier normal initialized weights

    """
    if isinstance(model, nn.Conv2d):
        xavier_normal_(model.weight.data)

def accuracy(data_iterator):
    """Short summary.

    Parameters
    ----------
    data_iterator : type
        Description of parameter `data_iterator`.

    Returns
    -------
    type
        Description of returned object.

    """
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator:
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1)==y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total/length
        return accuracy

class ConvNet(nn.Module):
    def __init__(self, OUTPUT):
        super(ConvNet, self).__init__()
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
