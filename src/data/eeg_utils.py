import mne
import numpy as np
import os
import seaborn as sb
import pandas as pd
from mne.parallel import parallel_func
from mne.decoding import CSP
from scipy import signal
from scipy.signal import resample
from datetime import datetime
from autoreject import Ransac, AutoReject
from mne.time_frequency import tfr_stockwell, psd_array_welch, psd_welch, psd_multitaper
from autoreject import get_rejection_threshold
import yaml

# Import configuration
config = yaml.load(open('./config.yml'))
epoch_length = config['epoch_length']


def get_eeg_path(subject, raw=True):
    """
    Get the EEG file path and Force data path of a subject

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    raw     : raw eeg file or decontaminated (by B-Alert) file

    Returns
    ----------
    eeg_file_path   : path to a EEG data to the subject
    """
    # EEG file
    eeg_path = '../data/raw/eeg_data/' + subject + '/'
    fname = [f for f in os.listdir(eeg_path) if f.endswith('.edf')]
    fname.sort()
    if raw:
        eeg_path = eeg_path + fname[1]  # raw file
    else:
        eeg_path = eeg_path + fname[0]  # Decontaminated file

    return eeg_path


def get_trial_path(subject, trial):
    """
    Get the trial file path  a subject

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    trial_file_path   : path to a trial (Force) data to the subject
    """
    # Trial time
    trial_path = '../data/raw/force_data/' + subject + '/'
    fname = [f for f in os.listdir(trial_path) if f.endswith('.csv')]
    fname.sort()
    idx = ['HighFine', 'HighGross', 'LowFine', 'LowGross'].index(trial)
    trial_path = trial_path + fname[idx]

    return trial_path


def get_eeg_time(subject):
    """
    Get the start time of eeg trial

    Parameter
    ----------
    subject   : path to eeg file

    Returns
    ----------
    start_time : start time of the eeg trial

    """
    # EEG time
    eeg_path = get_eeg_path(subject)
    eeg_time = eeg_path.split('.')
    eeg_time = datetime.strptime(
        ''.join(eeg_time[3:5]) + '0000', '%d%m%y%H%M%S%f')

    return eeg_time


def get_trial_time(subject, trial):
    """
    Get the start and end time of a trial to align with eeg data

    Parameter
    ----------
    subject  : path to eeg file

    Returns
    ----------
    trial_start : start time of the trial with eeg as reference
    trial_end   : end time of the trial with eeg as reference
    """
    # EEG time
    eeg_time = get_eeg_time(subject)
    trial_path = get_trial_path(subject, trial)

    # Trial time
    trial_time = np.genfromtxt(trial_path, dtype=str, delimiter=',',
                               usecols=0, skip_footer=150, skip_header=100).tolist()

    # Update year, month, and day
    start_t = datetime.strptime(trial_time[0], '%H:%M:%S:%f')
    start_t = start_t.replace(year=eeg_time.year,
                              month=eeg_time.month, day=eeg_time.day)
    end_t = datetime.strptime(trial_time[-1], '%H:%M:%S:%f')
    end_t = end_t.replace(year=eeg_time.year,
                          month=eeg_time.month, day=eeg_time.day)

    trial_start = (start_t - eeg_time).total_seconds()  # convert to seconds
    trial_end = (end_t - eeg_time).total_seconds()

    return trial_start, trial_end


def get_eeg_data(subject):
    """
    Get the eeg data excluding unnessary channels from edf file

    Parameter
    ----------
    subject : string of subject ID e.g. 7707

    Returns
    ----------
    raw_selected : selected raw eeg out of whole experiment
    """
    eeg_path = get_eeg_path(subject)
    eeg_time = get_eeg_time(subject)
    # EEG info
    info = mne.create_info(ch_names=['Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1',
                                     'P3', 'Pz', 'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2',
                                     'STI 014'],
                           ch_types=['eeg'] * 20 + ['stim'],
                           sfreq=256.0,
                           montage="standard_1020")
    # Read the raw data
    exclude = ['ECG', 'AUX1', 'AUX2', 'AUX3', 'ESUTimestamp',
               'SystemTimestamp', 'Tilt X', 'Tilt Y', 'Tilt Z']
    raw = mne.io.read_raw_edf(eeg_path, preload=True,
                              exclude=exclude, verbose=False)
    data = raw.get_data()
    raw_selected = mne.io.RawArray(data, info, verbose=False)

    # Additional information
    meas_date = eeg_time.strftime('%m-%d-%Y,%H:%M:%S')
    raw_selected.info['description'] = meas_date
    raw_selected.info['subject_info'] = subject
    raw_selected.info['experimenter'] = 'hemanth'

    return raw_selected


def create_epochs(subject, trial, preload=True):
    """
    Get the epcohed eeg data excluding unnessary channels from fif file and also filter the signal

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    epochs  : epoched data
    """
    trial_start, trial_end = get_trial_time(subject, trial)
    raw = get_eeg_data(subject)
    raw_cropped = raw.copy().crop(tmin=trial_start, tmax=trial_end)  # Crop the trials
    raw_cropped.notch_filter(60, filter_length='auto',
                             phase='zero', verbose=False)  # Line noise
    raw_cropped.filter(l_freq=1, h_freq=50, fir_design='firwin',
                       verbose=False)  # Band pass filter
    raw_cropped.set_eeg_reference('average')
    events = mne.make_fixed_length_events(raw_cropped, duration=epoch_length)
    epochs = mne.Epochs(raw_cropped, events, tmin=0,
                        tmax=epoch_length, verbose=False, preload=preload)

    return epochs


def autoreject_epochs(epochs, reject_plot=False):
    """
    Rejects the bad epochs with AutoReject algorithm

    Parameter
    ----------
    epochs : Epoched, filtered eeg data

    Returns
    ----------
    epochs : Epoched data after rejection of bad epochs

    """
    # Cleaning with autoreject
    picks = mne.pick_types(epochs.info, eeg=True)  # Pick EEG channels
    ar = AutoReject(n_interpolate=[1, 4, 8], n_jobs=6, picks=picks,
                    thresh_func='bayesian_optimization',
                    cv=10,
                    random_state=42, verbose=False)

    cleaned_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    if reject_plot:
        reject_log.plot_epochs(epochs, scalings=dict(eeg=40e-6))

    return cleaned_epochs


def run_ica(epochs, reject):
    """
    Runs ICA on the given epochs data

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data

    Returns
    ----------
    ICA : ICA object from mne

    """
    picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                           eog=False, stim=False, exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="picard", verbose=False)
    ica.fit(epochs, picks=picks, reject=reject)

    return ica


def find_eog(epochs, ica):
    """
    Detects the eye blink aritifact indices

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne

    Returns
    ----------
    ICA : ICA object with eog indices appended

    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp1', verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp2', verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    return ica
