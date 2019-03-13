from utils import *
import yaml
import torch
from networks import ShallowEEGNet
import deepdish as dd
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import CustomDataset


def predict(model_path, parameters):
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
        accuracy of classification.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_iterator = create_data_iterator(parameters, predicting=True)
    trained_model = torch.load(model_path, map_location=device)
    # trained_model['model'].eval()
    output = []
    with torch.no_grad():
        for x, y in data_iterator:
            output.append(trained_model(x.to(device)))

    return output


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
                  'data_path': str(data_path)}
    read_path = str(Path(__file__).parents[2] / 'models')
    with open(read_path + '/time.txt', "r+") as f:
        latest_model = f.readlines()[-1].splitlines()
    model_path = str(
        Path(__file__).parents[2] / 'models/model_') + latest_model[0] + '.pth'
    predictions = predict(model_path, parameters)
    print(predictions)
