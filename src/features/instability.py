import deepdish as dd
from scipy.signal import welch
from pathlib import Path
import numpy as np


def interaction_band_pow(subject, trial, config):
    """Get the band power (psd) of the interaction forces/moments.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.

    Returns
    -------
    band_pow : array
        An array of band_pow of each epoch of data.
    freqs : array
        An array of frequencies in power spectral density.

    """

    read_path = path = str(
        Path(__file__).parents[2] / config['raw_robot_dataset'])
    all_data = dd.io.load(path)
    data = all_data[subject]['robot'][trial].get_data()
    # Variables
    freq_bands = [[0, 128.0]]
    epochs = data.shape[0]
    idx = [
        'x', 'y', 'force_x', 'force_y', 'total_force', 'moment_x', 'moment_y',
        'total_moment', 'moment_scaled'
    ].index('moment_scaled')

    # Calculate the power
    band_pow = []
    for i in range(epochs):
        freqs, power = welch(data[i, idx, :],
                             fs=256,
                             nperseg=128,
                             nfft=256,
                             detrend=False)
        band_pow.append(power)
    band_pow = np.sqrt(np.array(band_pow) * 256 / 2)

    return band_pow, np.array(freqs)


def instability_index(subject, trial, config):
    """Calculate instability index of the subject and trial.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.
    config : yaml file
        The configuration file

    Returns
    -------
    ins_index : array
        An array of instability index calculated at each epochs.

    """

    # signature of data: x(n_epochs, 3 channels, frequencies: 0-128 Hz)
    data, freqs = interaction_band_pow(subject, trial, config)

    # Get frequency index between 3 Hz and 12 Hz
    f_min_max = (freqs <= 10) & (freqs > 2)
    f_0_max = (freqs <= 10) & (freqs > 0)
    num = data[:, f_min_max].sum(axis=1)
    den = data[:, f_0_max].sum(axis=1)
    ins_index = np.expand_dims(num / den, axis=1).flatten()

    return ins_index
