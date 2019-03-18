import yaml
import torch
import deepdish as dd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def plot_robot_position(subject, trial, config):
    """Plots the robot end effector position (only x and y are plotted).

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.
    config: yaml file
        configuration file with all parameters

    """
    # Read the data from processed data folder
    path = str(Path(__file__).parents[2] / config['raw_robot_dataset'])
    all_data = dd.io.load(path)
    temp_data = all_data[subject]['robot'][trial]
    n_features = len(temp_data.info['ch_names'])
    sub_data = temp_data.get_data().transpose(1, 0, 2).reshape(n_features, -1)
    # plotting
    plt.plot(sub_data[1, :], sub_data[0, :])  # robot co-ordinates are flipped
    plt.show()

    return None


def plot_model_accuracy(model_path):
    """Plot training, validation, and testing acurracy.

    Parameters
    ----------
    model_path : str
        A path to saved pytorch model.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(model_path, map_location=device)
    print(model_info)
    training_accuracy = model_info['training_accuracy']
    validation_accuracy = model_info['validation_accuracy']
    testing_accuracy = model_info['testing_accuracy']

    epochs = np.arange(training_accuracy.shape[0])

    sb.set()
    plt.plot(epochs, training_accuracy, epochs,
             validation_accuracy, epochs, testing_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    return None


def plot_predictions(subject, trial, config, predictions):
    """A function to plot trajectory and predictions along the path.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.
    predictions : numpy array
        A numpy array of (m x 1) of predictions at required points.

    """
    # Co-ordinates
    path = str(Path(__file__).parents[2] / config['raw_robot_dataset'])
    all_data = dd.io.load(path)
    temp_data = all_data[subject]['robot'][trial]
    sub_data = temp_data.get_data()[:, :, 128] # Select only middle point of the epoch
    x, y = sub_data[:,1], sub_data[:,0]
    if len(x)!=len(predictions): # check is number of predictions matach x or y data
        raise Exception('Two epochs are not of same length!')
    # Find three classes
    idx_up = np.where(predictions==0)
    idx_down = np.where(predictions==1)
    idx_O = np.where(predictions==2)
    plt.scatter(x[idx_up], y[idx_up], marker='^')
    plt.scatter(x[idx_O], y[idx_O], marker='o')
    plt.scatter(x[idx_down], y[idx_down], marker='v')
    plt.legend()
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()

    return None



if __name__ == '__main__':
    # Parameters
    config = yaml.load(open(str(Path(__file__).parents[1]) + '/config.yml'))
    subjects = config['subjects']
    trials = config['trials']

    plot_predictions(subjects[0], trials[0], config, predictions=[0, 1])
    # plot_model_accuracy(model_info_path)
