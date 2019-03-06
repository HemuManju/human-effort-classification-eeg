import deepdish as dd
from pathlib import Path
import pytest
import yaml

# Configuration files
path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(path))
epoch_length = config['epoch_length']
subjects = config['subjects']
trials = config['trials']

@pytest.mark.skip(reason='slow')
def test_eeg_data_length():
    """Short summary.
    Test the number of subjects in data

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    # Raw data
    path = str(Path(__file__).parents[1]/'data/interim/raw_eeg_dataset.h5')
    data = dd.io.load(path)
    assert (len(data.keys())==len(subjects))

    # Clean data
    path = str(Path(__file__).parents[1]/'data/processed/clean_eeg_dataset.h5')
    data = dd.io.load(path)
    assert (len(data.keys())==len(subjects))


@pytest.mark.skip(reason='slow')
def test_robot_data_length():
    """Short summary.
    Test the number of subjects in data

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    # Raw data
    path = str(Path(__file__).parents[1]/'data/processed/robot_dataset.h5')
    data = dd.io.load(path)
    assert (len(data.keys())==len(subjects))
