import yaml
import torch
import deepdish as dd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle


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
    training_accuracy = model_info['training_accuracy']
    validation_accuracy = model_info['validation_accuracy']
    testing_accuracy = model_info['testing_accuracy']
    epochs = np.arange(training_accuracy.shape[0])

    # Plotting
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
    # Select only middle point of the epoch
    sub_data = temp_data.get_data()[:, :, 128]
    x, y = sub_data[:, 1], sub_data[:, 0]
    if len(x) != len(predictions):  # check is number of predictions matach x or y data
        print(len(x), len(predictions))
        raise Exception('Two epochs are not of same length!')
    # Find three classes
    idx_up = np.where(predictions == 0)
    idx_down = np.where(predictions == 1)
    idx_O = np.where(predictions == 2)

    # Plotting
    sb.set()
    plt.scatter(x[idx_up], y[idx_up], marker='^')
    plt.scatter(x[idx_O], y[idx_O], marker='o')
    plt.scatter(x[idx_down], y[idx_down], marker='v')
    plt.tick_params(labelright=False, top=False,
                    labelleft=False, labelbottom=False)
    # plt.legend(['Increase', 'Hold', 'Decrease'])
    # plt.xlabel('x')
    # plt.ylabel('y')

    return None


def plot_predictions_with_instability(subject, trial, config, predictions, ins_index):
    """A function to plot trajectory and predictions along the path.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.
    predictions : numpy array
        A numpy array of (m x 1) of predictions at required points.
    ins_index : instability index
        A numpy array of (m x 1) of instability index at required points.

    """
    # Co-ordinates
    path = str(Path(__file__).parents[2] / config['raw_robot_dataset'])
    all_data = dd.io.load(path)
    temp_data = all_data[subject]['robot'][trial]
    # Select only middle point of the epoch
    sub_data = temp_data.get_data()[:, :, 128]
    x, y = sub_data[:, 1], sub_data[:, 0]
    if len(x) != len(predictions):  # check is number of predictions matach x or y data
        print(len(x), len(predictions))
        raise Exception('Two epochs are not of same length!')
    # Find three classes
    idx_up = np.where(predictions == 0)
    idx_down = np.where(predictions == 1)
    idx_O = np.where(predictions == 2)

    # Plotting
    sb.set()
    plt.scatter(x[idx_up], y[idx_up], marker='^', s=ins_index[idx_up]*500)
    plt.scatter(x[idx_O], y[idx_O], marker='o', s=ins_index[idx_O]*500)
    plt.scatter(x[idx_down], y[idx_down], marker='v', s=ins_index[idx_down]*500)
    plt.tick_params(labelright=False, top=False,
                    labelleft=False, labelbottom=False)
    # plt.legend(['Increase', 'Hold', 'Decrease'])
    # plt.xlabel('x')
    # plt.ylabel('y')

    return None
