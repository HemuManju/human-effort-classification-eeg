import deepdish as dd
from pathlib import Path
import deepdish
import pytest
import yaml
import h5py
import numpy as np

# Configuration files
path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(path))
epoch_length = config['epoch_length']
subjects = config['subjects']
trials = config['trials']


# @pytest.mark.skip(reason='slow')
def test_eeg_dataset_length():
    """Test the number of subjects in eeg dataset.

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    # Raw dataset
    path = str(Path(__file__).parents[1] / config['raw_eeg_dataset'])
    data = dd.io.load(path)
    assert (len(data.keys()) == len(subjects))

    # Clean dataset
    path = str(Path(__file__).parents[1] / config['clean_eeg_dataset'])
    data = dd.io.load(path)
    assert (len(data.keys()) == len(subjects))


# @pytest.mark.skip(reason='need to do')
def test_eeg_robot_dataset_length():
    """This verify whether correct number of epochs have been dropped from eeg and robot dataset.

    Returns
    -------
    Assertion with respect to the length of two datasets
    """
    # Parameters
    trials = config['trials']
    # Raw dataset
    eeg_path = str(Path(__file__).parents[1] / config['clean_eeg_dataset'])
    robot_path = str(Path(__file__).parents[1] / config['raw_robot_dataset'])

    eeg_dataset = dd.io.load(eeg_path)
    robot_dataset = dd.io.load(robot_path)
    for key in eeg_dataset.keys():
        for trial in trials:
            eeg_data = eeg_dataset[key]['eeg'][trial].get_data()
            robot_data = robot_dataset[key]['robot'][trial].get_data()
            assert (eeg_data.shape[0] == robot_data.shape[0])


# @pytest.mark.skip(reason='need to do')
def test_robot_dataset_length():
    """Test the number of subjects in robot dataset.

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    # Raw data
    path = str(Path(__file__).parents[1] / config['raw_robot_dataset'])
    data = dd.io.load(path)
    assert (len(data.keys()) == len(subjects))


# @pytest.mark.skip(reason='need to do')
def test_torch_dataset_length():
    """Test whether number of features and labels are same.

    Returns
    -------
    Assertion whether total features and labels are equal.
    """
    # Raw data
    path = str(Path(__file__).parents[1] / config['torch_dataset'])
    data = dd.io.load(path)
    n_features = data['features'].shape[0]
    n_labels = data['labels'].shape[0]

    assert (n_labels == n_features)


@pytest.mark.skip(reason='need to do')
def test_data_iterator_ids():
    """Short summary.

    Returns
    -------
    Assertion whether length of test id is equal to length of validate id

    """
    from src.models.utils import data_iterator_ids
    path = str(Path(__file__).parents[1] / config['torch_dataset'])
    ids_list = data_iterator_ids(path, test_size=0.15)

    # Check validation and testing data size
    assert abs(len(ids_list['validation']) - len(ids_list['testing'])) < 2


@pytest.mark.skip(reason='need to do')
def test_data_balance():
    """Test if data is balanced.
    Returns
    -------
        Assertion whether the chance is less than 35%

    """
    path = str(Path(__file__).parents[1] / config['balanced_torch_dataset'])
    data = dd.io.load(path)
    labels = np.array(data['labels'])
    sum = np.sum(labels, axis=0) / len(labels)

    assert (sum < 0.36).all()
