import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import deepdish as dd


class SubjectSpecificDataset(Dataset):
    """A subject specific dataset class.

    Parameters
    ----------
    x : array
        Input array of the .

    Attributes
    ----------
    length : int
        Length of the dataset.

    """

    def __init__(self, x_data):
        super(SubjectSpecificDataset, self).__init__()
        self.length = x_data.shape[0]
        self.data = x_data

    def __getitem__(self, index):
        # Convert to torch tensors
        x = torch.from_numpy(self.data[index, :, :]).type(torch.float32)
        return x

    def __len__(self):
        return self.length


class CollectiveDataset(Dataset):
    """All subject dataset class.

    Parameters
    ----------
    ids_list : list
        ids list of training or validation or traning data.

    Attributes
    ----------
    ids_list

    """

    def __init__(self, ids_list, data_path):
        super(CollectiveDataset, self).__init__()
        self.ids_list = ids_list
        self.path = str(data_path)

    def __getitem__(self, index):
        id = self.ids_list[index]
        # Read only specific data
        x = dd.io.load(self.path, group='/features', sel=dd.aslice[id, :, :])
        y = dd.io.load(self.path, group='/labels', sel=dd.aslice[id, :])
        # Convert to torch tensors
        x = torch.from_numpy(x).type(torch.float32)
        y = torch.from_numpy(y).type(torch.float32)

        return x, y

    def __len__(self):
        return len(self.ids_list)
