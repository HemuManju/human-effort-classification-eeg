import yaml
import torch
import deepdish as dd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

config = yaml.load(open(str(Path(__file__).parents[1]) + '/config.yml'))


def robot_position_plot(subject, trial):
    """Short summary.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine.

    """
    # Read the data from processed data folder
    path = str(Path(__file__).parents[2] / config['robot_dataset'])
    all_data = dd.io.load(path)
    sub_data = all_data[subject]['robot'][trial]
    n_features = len(sub_data.info['ch_names'])
    data = sub_data.get_data()
    sub_data = data.transpose(1, 0, 2).reshape(n_features, -1)
    # plotting
    plt.plot(sub_data[0, :], sub_data[1, :])
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


if __name__ == '__main__':
    # Parameters
    path = str(Path(__file__).parents[2] / 'models')
    with open(path + '/time.txt', "r+") as f:
        time_stamp = f.readlines()
    time_stamp = time_stamp[-1][0:-1]
    model_path = path + '/model_' + time_stamp + '.pth'
    model_info_path = path + '/model_info_' + time_stamp + '.pth'

    # robot_position_plot(subjects[0], trials[1])
    plot_model_accuracy(model_info_path)
