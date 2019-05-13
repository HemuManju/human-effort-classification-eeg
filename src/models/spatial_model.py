import numpy as np
import mne
from pathlib import Path
import deepdish as dd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split


def svm_tangent_space_classifier(config):
    """A tangent space classifier with svm.

    Parameters
    ----------
    config : yaml file
        The configuration file.

    Returns
    -------
    sklearn classifier
        Learnt classifier.

    """

    n_components = 3  # pick some components
    path = str(Path(__file__).parents[2] / config['balanced_torch_dataset'])
    features = dd.io.load(path, group='/features')
    labels = dd.io.load(path, group='/labels')
    y = dd.io.load(path, group='/data_index')

    # Train test split
    train_id, test_id, _, _ = train_test_split(y,
                                               y * 0,
                                               test_size=2 *
                                               config['TEST_SIZE'])
    # Training
    train_features = features[train_id, :, :]
    train_labels = np.argmax(labels[train_id, :], axis=1)
    print(np.sum(labels[train_id, :], axis=0) / len(labels[train_id, :]))
    # Testing (not used here)
    test_features = features[test_id, :, :]
    test_labels = np.argmax(labels[test_id, :], axis=1)

    # Construct sklearn pipeline
    clf = make_pipeline(Covariances(estimator='lwf'), TangentSpace(),
                        SVC(kernel='linear'))
    # cross validation
    clf.fit(train_features, train_labels)

    return clf


def svm_tangent_space_prediction(clf, subject, trial, config):
    """Predict from learnt tangent space classifier.

    Parameters
    ----------
    clf : sklearn classifier
        Learnt sklearn classifier.
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.
    config : yaml file
        The configuration file.

    Returns
    -------
    array
        Predicted labels from the model.

    """

    n_electrodes = config['n_electrodes']
    epoch_length = config['epoch_length']
    s_freq = config['s_freq']
    data_path = str(Path(__file__).parents[2] / config['clean_eeg_dataset'])
    data = dd.io.load(data_path, group='/' + subject)
    x = data['eeg'][trial].get_data()
    test_features = x[:, 0:n_electrodes, 0:epoch_length * s_freq]

    # Predictions
    predictions = clf.predict(test_features)

    return predictions
