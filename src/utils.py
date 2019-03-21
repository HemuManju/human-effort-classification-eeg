from decorators import skip
import yaml
import deepdish as dd
from pathlib import Path
import seaborn as sb
import torch
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.stats import mode
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import mode
from models.predict_model import predict_all_task, predict_subject_task_specific


def get_model_path(experiment):
    """Get all the trained model paths from experiment.

    Parameters
    ----------
    experiment : str
        Which experiment trained models to load.

    Returns
    -------
    model path and model info path

    """

    path = str(Path(__file__).parents[1] / 'models')
    with open(path + '/time.txt', "r+") as f:
        time_stamp = f.readlines()
    time_stamp = time_stamp[-1][0:-1]
    model_path = path + '/model_' + time_stamp + '.pth'
    model_info_path = path + '/model_info_' + time_stamp + '.pth'

    return model_path, model_info_path


def save_dataset(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    if save:
        dd.io.save(path, dataset)

    return None


def voted_labels(experiment, subject, trial, config):
    """Short summary.

    Parameters
    ----------
    experiment : str
        Experiment to use for .
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine.
    config : yaml file
        The configuration file.

    Returns
    -------
    array
        Voted labels from trained classifiers from experiment.

    """

    read_path = str(Path(__file__).parents[1]) + '/models/' + experiment
    with open(read_path + '/time.txt', "r+") as f:
        trained_models = f.readlines()
    # Voting
    labels = []
    for trained_model in trained_models:
        model_time = trained_model.splitlines()[0] # remove "\n"
        model_path = str(Path(__file__).parents[1]) + '/models/' + experiment + '/model_' + model_time + '.pth'
        # Predictions
        predicted_labels = predict_subject_task_specific(model_path,
                                                         subject, trial, config)
        # voting system
        labels.append(predicted_labels)
    vote, _ = mode(np.array(labels), axis=0)

    return vote[0]


def save_trained_pytorch_model(trained_model, trained_model_info, save_path):
    """Save pytorch model and info.

    Parameters
    ----------
    trained_model : pytorch model
    trained_model_info : dict
    save_path : str

    """

    time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    torch.save(trained_model, save_path
               + '/model_' + time_stamp + '.pth')
    torch.save(trained_model_info, save_path
               + '/model_info_' + time_stamp + '.pth')
    # Save time also
    with open(save_path + '/time.txt', "a") as f:
        f.write(time_stamp + '\n')

    return None
