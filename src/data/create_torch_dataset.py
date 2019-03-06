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


def convert_to_tensors(subject, trial):
    """Converts the edf files in eeg and robot dataset into tensors.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross.

    Returns
    -------
    tensors
        x and y tensors corresponding to the subject and trial.

    """
    eeg_path = str(Path(__file__).parents[2] /'data/interim/clean_eeg_dataset.h5')
    data = dd.io.load(eeg_path, group='/'+subject)
    x = data['eeg'][trial].get_data()
    if trial=='HighFine': category = [1, 0, 0]
    if trial=='LowGross': category = [0, 1, 0]
    if (trial=='HighGross') or (trial=='LowFine'): category = [0, 0, 1]
    print(category, trial=='HighFine')
    y = one_hot_encode(x.shape[0], category)

    x_tensor = torch.from_numpy(x[:,0:n_electrodes,0:epoch_length*s_freq])
    y_tensor = torch.from_numpy(y)

    return x_tensor.type(torch.float32), y_tensor.type(torch.float32)


def create_torch_dataset(subjects, trials):
    """Create pytorch dataset for all subjects.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross.

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the tensors
    torch_dataset = {}
    x_temp = torch.empty((0, n_electrodes, epoch_length*s_freq))
    y_temp = torch.empty((0, n_class))

    for subject in subjects:
        for trial in trials:
            x_tensor, y_tensor = convert_to_tensors(subject, trial)
            x_temp = torch.cat((x_temp, x_tensor), dim=0)
            y_temp = torch.cat((y_temp, y_tensor), dim=0)
    torch_dataset['features'] = x_temp
    torch_dataset['labels'] = y_temp

    return torch_dataset


if __name__ == '__main__':
    path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(path))
    subjects = config['subjects']
    trials = config['trials']
    s_freq = config['s_freq']
    epoch_length = config['epoch_length']
    n_electrodes = config['n_electrodes']
    n_class = config['n_class']

    # Main file
    torch_dataset = create_torch_dataset(subjects, trials)
    save = True  # Save the file
    if save:
        save_path = str(Path(__file__).parents[2] / \
            'data/processed/torch_dataset.h5')
        dd.io.save(save_path, torch_dataset)
