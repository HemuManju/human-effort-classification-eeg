from torch.nn.init import xavier_normal_
import deepdish as dd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import time
from torchnet.logger import VisdomPlotLogger


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
    """Calculate the classification accuracy of all data_iterators.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : dict
        A dictionary with different datasets.

    Returns
    -------
    list
        A dictionary of accuracy for all datasets.

    """
    accuracy = []
    keys = data_iterator.keys()
    for key in keys:
        accuracy.append(calculate_accuracy(model, data_iterator, key))

    return accuracy


def calculate_accuracy(model, data_iterator, key):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : pytorch object
        A pytorch dataset.
    key : str
        A key to select which dataset to evaluate

    Returns
    -------
    float
        accuracy of classification for the given key.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator[key]:
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1) == y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total / length

    return accuracy.numpy()


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


def visual_log(title):
    """Return a pytorch tnt visual loggger.

    Parameters
    ----------
    title : str
        A title to describe the logging.

    Returns
    -------
    type
        pytorch visual logger.

    """
    visual_logger = VisdomPlotLogger('line',
                                     opts=dict(legend=['Training', 'Validation', 'Testing'],
                                               xlabel='Epochs', ylabel='Accuracy', title=title))

    return visual_logger


def create_model_info(model, param, accuracy):
    """Create a dictionary of relevant model info.

    Parameters
    ----------
    model : pytorch object
        A trained model from pytorch.
    param : dict
        Any parameter relevant for logging.
    accuracy_log : dict
        A dictionary containing accuracies.

    Returns
    -------
    type
        Description of returned object.

    """
    model_info = {'model': model,
                  'model_parameters': param,
                  'training_accuracy': accuracy[:, 0],
                  'validation_accuracy': accuracy[:, 1],
                  'testing_accuracy': accuracy[:, 2]}

    return model_info
