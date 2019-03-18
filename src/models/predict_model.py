from utils import *
import yaml
import torch
from networks import ShallowEEGNet
import deepdish as dd
from torch.utils.data import DataLoader
from pathlib import Path
import collections
import pickle


def predict(trained_model_path, config, subject_specific):
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
    labels = collections.defaultdict(dict)
    for subject in config['subjects']:
        for trial in config['trials']:
            if subject_specific:
                data_iterator = subject_specific_data_iterator(subject, trial)
                labels[subject][trial] = calculate_predictions(
                    trained_model, data_iterator, config)
            else:
                data_iterator = collective_data_iterator(config)
                labels[subject][trial] = calculate_predictions(
                    trained_model, data_iterator, config)
    return labels


if __name__ == '__main__':
    read_path = str(Path(__file__).parents[2] / 'models')
    with open(read_path + '/time.txt', "r+") as f:
        latest_model = f.readlines()[-1].splitlines()[0]
    model_path = str(
        Path(__file__).parents[2] / 'models/model_') + latest_model + '.pth'
    # Predictions
    predicted_labels = predict(model_path, config, subject_specific=True)
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(predicted_labels, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print(predicted_labels)
