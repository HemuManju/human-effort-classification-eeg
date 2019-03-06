import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import deepdish as dd
import yaml


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
    path = str(Path(__file__).parents[2] / 'data/processed/robot_dataset.h5')
    all_data = dd.io.load(path)
    sub_data = all_data[subject]['robot'][trial]
    n_features = len(sub_data.info['ch_names'])
    data = sub_data.get_data()
    sub_data = data.transpose(1, 0, 2).reshape(n_features, -1)
    # plotting
    plt.plot(sub_data[0, :], sub_data[1, :])
    plt.show()
    return None


if __name__ == '__main__':
    # Import configuration
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))
    epoch_length = config['epoch_length']
    subjects = config['subjects']
    trials = config['trials']

    robot_position_plot(subjects[0], trials[1])
