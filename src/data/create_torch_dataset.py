import deepdish as dd
import torch
import yaml
from pathlib import Path
import deepdish as dd
import numpy as np


def one_hot_encode(label_length, category):
    """Generate one hot encoded value of required length and category.

    Parameters
    ----------
    label_length : int
        required lenght of the array.
    category : int
        Caterory e.g: category=2, [0, 1, 0] in 3 class system

    Returns
    -------
    array
        One hot encoded array.

    """
    y = np.zeros((label_length, len(category)))
    y[:, category.index(1)] = 1

    return y


def convert_to_array(subject, trial):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    tensors
        x and y arrays corresponding to the subject and trial.

    """
    eeg_path = str(
        Path(__file__).parents[2] / config['clean_eeg_dataset'])
    data = dd.io.load(eeg_path, group='/' + subject)
    x = data['eeg'][trial].get_data()
    if trial == 'HighFine':
        category = [1, 0, 0]
    if trial == 'LowGross':
        category = [0, 1, 0]
    if (trial == 'HighGross') or (trial == 'LowFine'):
        category = [0, 0, 1]

    x_array = x[:, 0:n_electrodes, 0:epoch_length * s_freq]
    # In order to accomodate testing
    try:
        y_array = one_hot_encode(x.shape[0], category)
    except:
        y_array = np.zeros((x.shape[0], 3))

    return x_array, y_array


def torch_dataset(subjects, trials, config):
    """Create pytorch dataset for all subjects.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array
    torch_dataset = {}
    x_temp = np.empty((0, config['n_electrodes'], config['epoch_length'] * config['s_freq']))
    y_temp = np.empty((0, config['n_class']))

    for subject in subjects:
        for trial in trials:
            x_array, y_array = convert_to_array(subject, trial)
            x_temp = np.concatenate((x_temp, x_array), axis=0)
            y_temp = np.concatenate((y_temp, y_array), axis=0)
    torch_dataset['features'] = x_temp
    torch_dataset['labels'] = y_temp
    torch_dataset['data_index'] = np.arange(y_temp.shape[0])

    return torch_dataset


def balanced_torch_dataset(data_path):
    """Create balanced pytorch dataset for all subjects.

    Parameters
    ----------
    data_path : str
        Path to pytroch dataset.

    Returns
    -------
    dict
        A balanced dataset.

    """
    balanced_dataset = {}
    data = dd.io.load(data_path)
    features = np.array(data['features'])
    labels = np.array(data['labels'])
    ids = data['data_index']
    # Get each class labels
    class_1_ids = ids[np.argmax(labels, axis=1) == 0]
    class_2_ids = ids[np.argmax(labels, axis=1) == 1]
    class_3_ids = ids[np.argmax(labels, axis=1) == 2]

    # Drop 50% of class 3 labels
    index = np.random.choice(
        class_3_ids.shape[0], int(class_3_ids.shape[0] / 2), replace=False)
    # print(sum(kept) / len(ids))
    class_3_ids = class_3_ids[index]
    # Concatenate all of them
    ids_list = np.hstack((class_1_ids, class_2_ids, class_3_ids))
    balanced_dataset['features'] = features[ids_list, :, :]
    balanced_dataset['labels'] = labels[ids_list, :]
    balanced_dataset['data_index'] = np.arange(ids_list.shape[0])

    return balanced_dataset
