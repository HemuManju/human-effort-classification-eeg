from robot_utils import *
import deepdish as dd
from pathlib import Path
import yaml


def create_dataset(subjects, trials):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    robot_dataset : dataset of all the subjects with different conditions.

    """
    robot_dataset = {}
    for subject in subjects:
        data = {'robot': {'HighFine': None, 'HighGross': None,
                          'LowFine': None, 'LowGross': None}}
        for trial in trials:
            data['robot'][trial] = create_robot_epochs(subject, trial)
        robot_dataset[subject] = data

    return robot_dataset


if __name__ == '__main__':
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))
    subjects = config['subjects']
    trials = config['trials']

    # Main file
    robot_dataset = create_dataset(subjects, trials)
    save = True  # Save the file
    if save:
        save_path = Path(__file__).parents[2] / \
            'data/interim/robot_dataset.h5'
        dd.io.save(save_path, robot_dataset)
