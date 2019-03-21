from torch.nn.init import xavier_normal_
import deepdish as dd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import time
from torchnet.logger import VisdomPlotLogger
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from .datasets import CollectiveDataset, SubjectSpecificDataset


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


def collective_data_iterator(config, predicting=False):
    """Create data iterators.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    parameters: dict
        A dictionary of parameters

    Returns
    -------
    dict
        A dictionary contaning traning, validation, and testing iterator.

    """
    data_iterator = {}
    data_path = str(
        Path(__file__).parents[2] / config['balanced_torch_dataset'])
    BATCH_SIZE = config['BATCH_SIZE']
    TEST_SIZE = config['TEST_SIZE']
    data = dd.io.load(data_path)
    if predicting:
        ids_list = dd.io.load(data_path, group='/data_index')
        # Create datasets
        test_data = CollectiveDataset(ids_list, data_path)
        # Load datasets
        data_iterator = DataLoader(test_data, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=10)
    else:
        # Load datasets
        ids_list = data_iterator_ids(data_path, test_size=TEST_SIZE)
        # Create datasets
        train_data = CollectiveDataset(ids_list['training'], data_path)
        valid_data = CollectiveDataset(ids_list['validation'], data_path)
        test_data = CollectiveDataset(ids_list['testing'], data_path)
        # Data iterators
        data_iterator['training'] = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
        data_iterator['validation'] = DataLoader(
            valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
        data_iterator['testing'] = DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)

    return data_iterator


def subject_specific_data_iterator(subject, trial, config):
    """A subject specific data iterator.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    pytorch dataiterator
        A pytorch data iterator

    """

    # Parameters
    n_electrodes = config['n_electrodes']
    epoch_length = config['epoch_length']
    s_freq = config['s_freq']
    data_path = str(
        Path(__file__).parents[2] / config['clean_eeg_dataset'])
    data = dd.io.load(data_path, group='/' + subject)
    x = data['eeg'][trial].get_data()
    x = x[:, 0:n_electrodes, 0:epoch_length * s_freq]
    # Create datasets
    test_data = SubjectSpecificDataset(x)
    # Load datasets
    data_iterator = DataLoader(test_data, batch_size=x.shape[0],
                               shuffle=False, num_workers=10)

    return data_iterator


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


def calculate_predictions(trained_model, data_iterator, parameters):
    """Calculate the predictions from the model, .

    Parameters
    ----------
    trained_model : pytorch object
        A trained pytorch model.
    data_iterator : pytorch object
        A pytorch dataset.
    subject : str
        A key to select which dataset to evaluate

    Returns
    -------
    Array
        Labels for the given subject and dataset.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = np.empty((0, parameters['OUTPUT']))
    with torch.no_grad():
        for x in data_iterator:
            temp = trained_model(x.to(device)).cpu().detach()
            output = np.concatenate((output, temp.numpy()), axis=0)
    predicted_labels = np.argmax(output, axis=1)

    return predicted_labels


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


def create_model_info(config, loss_func, accuracy):
    """Create a dictionary of relevant model info.

    Parameters
    ----------
    param : dict
        Any parameter relevant for logging.
    accuracy_log : dict
        A dictionary containing accuracies.

    Returns
    -------
    type
        Description of returned object.

    """
    model_info = {'training_accuracy': accuracy[:, 0],
                  'validation_accuracy': accuracy[:, 1],
                  'testing_accuracy': accuracy[:, 2],
                  'model_parameters': config,
                  'loss function': loss_func}

    return model_info
