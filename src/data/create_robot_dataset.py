import deepdish as dd
from pathlib import Path
import yaml
import collections
from .robot_utils import *


def robot_dataset(subjects, trials):
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
        data = collections.defaultdict(dict)
        for trial in trials:
            data['robot'][trial] = create_robot_epochs(subject, trial)
        robot_dataset[subject] = data

    return robot_dataset
