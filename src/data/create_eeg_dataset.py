from eeg_utils import *
import deepdish as dd
import yaml

def create_dataset(subjects, trials):
    """
    create the data with each subject data in a dictionary
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
        data = {'eeg':{'HighFine': None, 'HighGross': None,
        'LowFine': None,'LowGross': None}}
        for trial in trials:
            data['eeg'][trial] = create_epochs(subject, trial)
        eeg_dataset[subject] = data

    return eeg_dataset


if __name__=='__main__':
    config = yaml.load(open('./config.yml'))
    subjects = config['subjects']
    trials = config['trials']

    eeg_dataset = create_dataset(subjects, trials)
    save=True # Save the file
    if save:
        dd.io.save('../data/processed/eeg_raw.h5', eeg_dataset)
