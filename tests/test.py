import deepdish as dd
from pathlib import Path
import deepdish
import pytest
import yaml
import h5py

# Configuration files
path = Path(__file__).parents[1]/'src/config.yml'
config = yaml.load(open(path))
epoch_length = config['epoch_length']
subjects = config['subjects']
trials = config['trials']

@pytest.mark.skip(reason='slow')
def test_eeg_data_length():
    """Test the number of subjects in eeg dataset.

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    # Raw data
    path = str(Path(__file__).parents[1]/'data/interim/raw_eeg_dataset.h5')
    data = dd.io.load(path)
    assert (len(data.keys())==len(subjects))

    # Clean data
    path = str(Path(__file__).parents[1]/'data/interim/clean_eeg_dataset.h5')
    data = dd.io.load(path)
    assert (len(data.keys())==len(subjects))


@pytest.mark.skip(reason='slow')
def test_robot_data_length():
    """Test the number of subjects in robot dataset.

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    # Raw data
    path = str(Path(__file__).parents[1]/'data/interim/robot_dataset.h5')
    data = dd.io.load(path)
    assert (len(data.keys())==len(subjects))


def test_torch_data_length():
    """Test whether number of features and labels are same.

    Returns
    -------
    Assertion with respect to the total features and labels.
    """
    # Raw data
    path = str(Path(__file__).parents[1]/'data/processed/torch_dataset.h5')
    data = dd.io.load(path)
    n_features = data['features'].shape[0]
    n_labels = data['labels'].shape[0]
    assert (n_labels==n_features)
