from parameters.py import *
from surface_laplacian import surface_laplacian
import mne
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from dateutil import tz
from pytz import timezone
from mne.parallel import parallel_func
import sys
sys.path.append('./')

# Set some save path parameters depending on epoch length
if epoch_length > 1:
    save_path = '/Cleaned_EEG/'
    read_path = '/Raw_EEG/'
else:
    save_path = '/Cleaned_EEG_High_Res/'
    read_path = '/Raw_EEG_High_Res/'


def get_file_path(subject):
    # EEG file
    eeg_path = '../EEG Data/' + subject + '/'
    fname = [f for f in os.listdir(eeg_path) if f.endswith('.edf')]
    fname.sort()
    eeg_file_path = eeg_path + fname[1]  # Decontaminated file

    # Trial time
    trial_path = '../Force Data/' + subject + '/'
    fname = [f for f in os.listdir(trial_path) if f.endswith('.csv')]
    fname.sort()
    # Four files: 'HighFine', 'HighGross', 'LowFine', 'LowGross'
    trial_file_path = [trial_path + path for path in fname[0:4]]

    return eeg_path, eeg_file_path, trial_path, trial_file_path


def clean_with_ica(subject):
    # get the path
    eeg_path, eeg_file_path, trial_path, trial_file_path = get_file_path(
        subject)

    trials = ['HighFine', 'HighGross', 'LowFine',
              'LowGross']  # Different trials
    for trial in trials:
        # Load the epoched data and ica
        epochs = mne.read_epochs(eeg_path + read_path + subject + \
                                 '_' + trial + '_epo.fif', verbose=False)  # read the trial
        ica = mne.preprocessing.read_ica(
            eeg_path + '/ICA_Results/' + subject + '_' + trial + '_ica' + '.fif')  # read th ica
        ica.exclude = []
        ica.plot_components(inst=epochs, cmap='viridis')
        ica.apply(epochs)
        epochs.save(eeg_path + save_path + subject
                    + '_' + trial + '_cleaned_epo.fif')
        ica.save(eeg_path + '/ICA_Results/'
                 + subject + '_' + trial + '_ica' + '.fif')
        print(ica.exclude)

    return 0


subjects = ['7707', '7708', '8801', '8802', '8803',
            '8815', '8819', '8820', '8821', '8822', '8823', '8824']

# Enable CUDA
parallel, run_func, _ = parallel_func(clean_with_ica, n_jobs=6)
parallel(run_func(subject) for subject in subjects)
