import yaml
import torch
import deepdish as dd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from .utils import get_model_path, figure_asthetics


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
    fig, ax = plt.subplots()
    ax.plot(sub_data[1, :], sub_data[0, :], alpha=0.35)  # robot co-ordinates are flipped
    plt.show()

    return None


def plot_average_model_accuracy(experiment, config):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fig, ax = plt.subplots()

    keys = ['training_accuracy', 'validation_accuracy', 'testing_accuracy']
    colors = [[0.69, 0.18, 0.45, 1.00], [0.49, 0.49, 0.49, 1.00], [0.12, 0.27, 0.59, 1.00]]
    for i, key in enumerate(keys):
        accuracy = np.empty((0, config['NUM_EPOCHS']))
        for j in range(5):
            model_path, model_info_path = get_model_path(experiment, j)
            model_info = torch.load(model_info_path, map_location=device)
            accuracy = np.vstack((model_info[key], accuracy))
        print(accuracy)
        average = np.mean(accuracy, axis=0)
        min_val = average - np.min(accuracy, axis=0)
        max_val = np.max(accuracy, axis=0) - average
        ax.fill_between(range(config['NUM_EPOCHS']), average-min_val, average+max_val, alpha=0.25, color=colors[i], edgecolor = colors[i])
        ax.plot(range(config['NUM_EPOCHS']), average, color = colors[i], label=key)

    ax.set_ylim(top=1.0)
    # Specifications
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    figure_asthetics(ax)
    plt.show()

    return None


def plot_model_accuracy(experiment, config, model_number):
    """Plot training, validation, and testing acurracy.

    Parameters
    ----------
    model_path : str
        A path to saved pytorch model.

    """

    model_path, model_info_path = get_model_path(experiment, model_number)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(model_info_path, map_location=device)
    training_accuracy = model_info['training_accuracy']
    validation_accuracy = model_info['validation_accuracy']
    testing_accuracy = model_info['testing_accuracy']
    epochs = np.arange(training_accuracy.shape[0])
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(epochs, training_accuracy, color=[0.69, 0.18, 0.45, 1.00])
    ax.plot(epochs, validation_accuracy, color=[0.69, 0.69, 0.69, 1.00])
    ax.plot(epochs, testing_accuracy, color=[0.12, 0.27, 0.59, 1.00])
    ax.set_ylim(top=1.0)
    # Specifications
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    figure_asthetics(ax)
    plt.show()

    return None


def plot_predictions(subject, trial, config, predictions, ins_index):
    """A function to plot trajectory and predictions.

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

    fig, ax = plt.subplots()
    if ins_index is None:
        # Plotting the prediction
        ax.scatter(x[idx_up], y[idx_up], marker='^')
        ax.scatter(x[idx_O], y[idx_O], marker='o')
        ax.scatter(x[idx_down], y[idx_down], marker='v')
    else:
        ax.scatter(x[idx_up], y[idx_up], marker='^', s=ins_index[idx_up] * 100)
        ax.scatter(x[idx_O], y[idx_O], marker='o', s=ins_index[idx_O] * 100)
        ax.scatter(x[idx_down], y[idx_down], marker='v',s=ins_index[idx_down] * 100)
    # Plotting the curve
    plot_robot_position(subject, trial, config)
    ax.legend(['Increase', 'Hold', 'Decrease'])
    ax.xlabel('x')
    ax.ylabel('y')
    plt.show()

    return None
