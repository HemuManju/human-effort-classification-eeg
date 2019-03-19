from .utils import *
import yaml
import torch
from .networks import ShallowEEGNet
import deepdish as dd
from torch.utils.data import DataLoader
from pathlib import Path
import collections
import pickle


def predict_all_task(trained_model_path, config, subject_specific):
    """Predict.

    Parameters
    ----------
    model_path : str
        Description of parameter `model_path`.
    config : dict
        A dictionary of hyper-parameters used in the network.

    Returns
    -------
    float
        Predicted labels from the model.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model = torch.load(trained_model_path, map_location=device)
    labels = collections.defaultdict(dict)
    for subject in config['subjects']:
        for trial in config['trials']:
            if subject_specific:
                data_iterator = subject_specific_data_iterator(subject, trial, config)
                labels[subject][trial] = calculate_predictions(
                    trained_model, data_iterator, config)
            else:
                data_iterator = collective_data_iterator(config)
                labels[subject][trial] = calculate_predictions(
                    trained_model, data_iterator, config)
    return labels


def predict_subject_task_specific(trained_model_path, subject, trial, config):
    """Predict subject and task specific labels.

    Parameters
    ----------
    trained_model_path : str
        Description of parameter `model_path`.
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.

    Returns
    -------
    array
        Predicted labels from the model.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model = torch.load(trained_model_path, map_location=device)
    data_iterator = subject_specific_data_iterator(subject, trial, config)
    labels = calculate_predictions(trained_model, data_iterator, config)

    return labels
