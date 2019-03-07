from torch.nn.init import xavier_normal_
import deepdish as dd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import datetime


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


def classification_accuracy(model, data_iterator):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : pytorch object
        A pytorch dataset.

    Returns
    -------
    float
        accuracy of classification.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator:
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1) == y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total / length

    return accuracy


def data_iterator_ids(path, test_size=0.15):
    """Generators training, validation, and training ids to be used by Dataloader.

    Parameters
    ----------
    path : str
        Path to the dataset.
    test_size : float
        Test size e.g. 0.15 is 15% of whole data.

    Returns
    -------
    dict
        A dictionary of ids corresponding to train, validate, and test.

    """
    y = dd.io.load(path, group='/data_index')
    ids_list = {}
    train_id, test_id, _, _ = train_test_split(
        y, y * 0, test_size=2 * test_size)
    test_id, validate_id, _, _ = train_test_split(
        test_id, test_id * 0, test_size=0.5)

    ids_list['training'] = train_id
    ids_list['validation'] = validate_id
    ids_list['testing'] = test_id

    return ids_list


def create_model_info(model):

    model_info = {'epoch': epoch,
                  'model': model,
                  'time': datetime.datetime.now(),
                  'training_accuracy': accuracy['training'],
                  'validation_accuracy': accuracy['validation']
                  'loss': loss}

    return model_info
