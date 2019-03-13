from utils import *
import yaml
import torch
from networks import ShallowEEGNet
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


if __name__ == '__main__':
    # Parameters
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))

    # Path to data to be tested
    data_path = Path(__file__).parents[2] / config['balanced_torch_dataset']

    # Paramters
    parameters = {'OUTPUT': config['OUTPUT'],
                  'NUM_EPOCHS': config['NUM_EPOCHS'],
                  'BATCH_SIZE': config['BATCH_SIZE'],
                  'LEARNING_RATE': config['LEARNING_RATE'],
                  'TEST_SIZE': config['TEST_SIZE'],
                  'subjects': config['subjects'][0:1],
                  'trials': config['trials'][0:1],
                  'data_path': str(data_path)}

    read_path = str(Path(__file__).parents[2] / 'models')
    with open(read_path + '/time.txt', "r+") as f:
        latest_model = f.readlines()[-1].splitlines()[0]
    model_path = str(
        Path(__file__).parents[2] / 'models/model_') + latest_model + '.pth'
    # Predictions
    predicted_labels = predict(model_path, parameters)
    print(predicted_labels.shape)
