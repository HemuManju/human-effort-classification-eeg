import deepdish as dd
import yaml
import collections
from .eeg_utils import *


def eeg_dataset(subjects, trials):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    eeg_dataset : dataset of all the subjects with different conditions

    """
    eeg_dataset = {}
    for subject in subjects:
        data = collections.defaultdict(dict)
        for trial in trials:
            data['eeg'][trial] = create_eeg_epochs(subject, trial)
        eeg_dataset[subject] = data

    return eeg_dataset
