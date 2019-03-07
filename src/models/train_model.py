from visdom import Visdom
from utils import weights_init, data_iterator_ids
import yaml
import torch
from networks import ShallowEEGNet
from torch.utils.data import DataLoader
import torch.nn as nn
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
    dict
        A dictionary contaning traning, validation, and testing iterator.

    """

    ids_list = data_iterator_ids(data_path, test_size=TEST_SIZE)

    # Create datasets
    train_data = CustomDataset(ids_list['training'])
    valid_data = CustomDataset(ids_list['validation'])
    test_data = CustomDataset(ids_list['testing'])

    # Load datasets
    data_iterator = {}
    data_iterator['training'] = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    data_iterator['validation'] = DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    data_iterator['testing'] = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    return data_iterator


def train(network, parameters, new_weights=False):
    """Main function to run the optimization..

    Parameters
    ----------
    network : class
        A pytorch network class.
    OUTPUT : int
        Number of classes.
    new_weights : bool
        Whether to use new weight initialization instead of default.

    Returns
    -------
    pytorch model
        A trained pytroch model.

    """
    # Device to train the model cpu or gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device being used:', device)

    data_iterator = create_data_iterator(parameters['data_path'],
                                         parameters['BATCH_SIZE'], parameters['TEST_SIZE'])

    # An instance of model
    model = network(parameters['OUTPUT']).to(device)
    if new_weights:
        model.apply(weights_init)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters['LEARNING_RATE'])

    for epoch in range(parameters['NUM_EPOCHS']):
        for x_batch, y_batch in data_iterator['training']:
            # Send the input and labels to gpu
            x_batch = x_batch.to(device)
            y_batch = (torch.max(y_batch, dim=1)[1]).to(device)

            # Forward pass
            out_put = model(x_batch)
            loss = criterion(out_put, y_batch)

            # Backward and optimize
            optimizer.zero_grad()  # For batch gradient optimisation
            loss.backward()
            optimizer.step()
        print(epoch)

    return model


if __name__ == '__main__':
    # Parameters
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))

    # Path to data
    data_path = Path(__file__).parents[2] / 'data/processed/torch_dataset.h5'

    # Network parameters
    parameters = {'OUTPUT': config['OUTPUT'],
                  'NUM_EPOCHS': config['NUM_EPOCHS'],
                  'BATCH_SIZE': config['BATCH_SIZE'],
                  'LEARNING_RATE': config['LEARNING_RATE'],
                  'TEST_SIZE': config['TEST_SIZE'],
                  'data_path': str(data_path)}

    trained_model = train(ShallowEEGNet, parameters)
    save = True
    save_path = str(Path(__file__).parents[2] / 'models')
    if save:
        torch.save(trained_model, save_path + '/trained_model.pth')
