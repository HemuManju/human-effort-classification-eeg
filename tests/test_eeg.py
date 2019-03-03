import deepdish as dd
import os
import pytest
import yaml

base_dir =  os.path.abspath(os.path.join(__file__ ,"../.."))
config = yaml.load(open(base_dir + '/src/config.yml'))
subjects = config['subjects']
trials = config['trials']

# @pytest.mark.skip(reason='slow')
def test_eeg_length():
    """Short summary.
    Test the number of subjects in data

    Returns
    -------
    Assertion with respect to the total number of subjects
    """
    fname = os.path.join(base_dir, 'data/processed/eeg_raw.h5')
    data = dd.io.load(fname)

    assert (len(data.keys())==len(subjects))
