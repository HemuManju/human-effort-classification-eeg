from eeg_utils import *
import deepdish as dd
import yaml
import collections


def create_dataset(subjects, trials):
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


if __name__ == '__main__':
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))
    subjects = config['subjects']
    trials = config['trials']
    # Main file
    eeg_dataset = create_dataset(subjects, trials)
    save = True  # Save the file
    if save:
        save_path = Path(__file__).parents[2] / \
            'data/interim/raw_eeg_exp_2_dataset.h5'
        dd.io.save(save_path, eeg_dataset)
