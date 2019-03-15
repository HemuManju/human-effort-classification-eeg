from .utils import *
import yaml
import torch
from .networks import ShallowEEGNet
import deepdish as dd
from torch.utils.data import DataLoader
from pathlib import Path


def predict(trained_model_path, parameters, subject_specific=False):
    """Predict.

    Parameters
    ----------
    model_path : str
        Description of parameter `model_path`.
    parameters : dict
        A dictionary of hyper-parameters used in the network.

    Returns
    -------
    float
        Predicted labels from the model.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model = torch.load(trained_model_path, map_location=device)
    for subject in parameters['subjects']:
        for trial in parameters['trials']:
            data_iterator = subject_specific_data_iterator(subject, trial)
            labels = calculate_predictions(
                trained_model, data_iterator, parameters)

    return labels
