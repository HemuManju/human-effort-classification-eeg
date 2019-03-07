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
    print(BATCH_SIZE)
    ids_list = dd.io.load(data_path, group='/data_index')
    # Create datasets
    train_data = CustomDataset(ids_list)
    # Load datasets
    data_iterator = DataLoader(train_data, batch_size=BATCH_SIZE,
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
    trained_model = torch.load(model_path)
    trained_model.eval()
    accuracy = classification_accuracy(trained_model, data_iterator)

    return accuracy


if __name__=='__main__':
    # Parameters
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))

    # Path to data to be tested
    data_path = Path(__file__).parents[2] / 'data/processed/torch_dataset.h5'

    # Paramters
    parameters = {'OUTPUT': config['OUTPUT'],
                  'NUM_EPOCHS': config['NUM_EPOCHS'],
                  'BATCH_SIZE': config['BATCH_SIZE'],
                  'LEARNING_RATE': config['LEARNING_RATE'],
                  'TEST_SIZE': config['TEST_SIZE'],
                  'data_path': str(data_path)}

    model_path = str(Path(__file__).parents[2] / 'models/trained_model.pth')
    accuracy = predict(model_path, parameters)

    print(accuracy)
