from utils import classification_accuracy, data_iterator_ids
import yaml
import torch
from networks import ShallowEEGNet
import deepdish as dd
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import CustomDataset


def create_data_iterator(data_path, BATCH_SIZE, TEST_SIZE):
    """Create data iterators.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    BATCH_SIZE : int
        Batch size of the data.
    TEST_SIZE : float
        Test size e.g 0.3 is 30% of the test data.

    Returns
    -------
    pytorch object
        A dataset iterator.

    """
    ids_list = dd.io.load(data_path, group='/data_index')
    # Create datasets
    test_data = CustomDataset(ids_list)
    data_iterator = {}
    # Load datasets
    data_iterator = DataLoader(test_data, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=6)

    return data_iterator


def predict(model_path, parameters):
    """Short summary.

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
    data_iterator = create_data_iterator(parameters['data_path'],
                                         parameters['BATCH_SIZE'], parameters['TEST_SIZE'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model = torch.load(model_path,map_location=device)
    model = trained_model['model'].eval()
    with torch.no_grad():
        prediction = []
        for x, y in data_iterator:
            output = model(x.to(device))
            prediction.append(output.cpu().detach())

    return prediction


if __name__ == '__main__':
    # Parameters
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))

    # Path to data to be tested
    data_path = Path(__file__).parents[2] / \
        'data/processed/balanced_torch_dataset.h5'

    # Paramters
    parameters = {'OUTPUT': config['OUTPUT'],
                  'NUM_EPOCHS': config['NUM_EPOCHS'],
                  'BATCH_SIZE': config['BATCH_SIZE'],
                  'LEARNING_RATE': config['LEARNING_RATE'],
                  'TEST_SIZE': config['TEST_SIZE'],
                  'data_path': str(data_path)}

    model_path = str(
        Path(__file__).parents[2] / 'models/model_Thu Mar  7 13:02:23 2019.pth')
    prediction = predict(model_path, parameters)
    print(prediction)
