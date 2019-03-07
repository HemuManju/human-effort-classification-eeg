from visdom import Visdom, server
from utils import *
import yaml
import torch
from networks import ShallowEEGNet
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from datasets import CustomDataset


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

    data_iterator = create_data_iterator(parameters)

    # An instance of model
    model = network(parameters['OUTPUT']).to(device)
    if new_weights:
        model.apply(weights_init)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters['LEARNING_RATE'])

    # Visual logger
    visual_logger = visual_log('Task type classification')
    accuracy_log = []
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

        accuracy = classification_accuracy(model, data_iterator)
        accuracy_log.append(accuracy)
        visual_logger.log(epoch, [accuracy[0], accuracy[1], accuracy[2]])

    # Add loss function info to parameter.
    parameters['loss_function'] = str(criterion)
    trained_model_info = create_model_info(
        model, parameters, np.array(accuracy_log))

    return trained_model_info


if __name__ == '__main__':
    # Parameters
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))

    # Path to data
    data_path = Path(__file__).parents[2] / \
        'data/processed/balanced_torch_dataset.h5'

    # Network parameters
    parameters = {'OUTPUT': config['OUTPUT'],
                  'NUM_EPOCHS': config['NUM_EPOCHS'],
                  'BATCH_SIZE': config['BATCH_SIZE'],
                  'LEARNING_RATE': config['LEARNING_RATE'],
                  'TEST_SIZE': config['TEST_SIZE'],
                  'data_path': str(data_path)}

    trained_model_info = train(ShallowEEGNet, parameters)
    save = True
    save_path = str(Path(__file__).parents[2] / 'models')
    if save:
        time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        torch.save(trained_model_info, save_path +
                   '/model_' + time_stamp + '.pth')
