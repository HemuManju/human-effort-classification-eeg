import sys

import numpy as np
import deepdish as dd
from pathlib import Path

import torch

from scipy.stats import mode
from datetime import datetime
from models.predict_model import predict_subject_task_specific
from contextlib import contextmanager


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """
    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>12}'.format(
                'Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>12}'.format('Running the block',
                                                       '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


def get_model_path(experiment, model_number):
    """Get all the trained model paths from experiment.

    Parameters
    ----------
    experiment : str
        Which experiment trained models to load.

    Returns
    -------
    model path and model info path

    """

    read_path = str(Path(__file__).parents[1]) + '/models/' + experiment
    with open(read_path + '/time.txt', "r+") as f:
        trained_models = f.readlines()[model_number]
    model_time = trained_models.splitlines()[0]  # remove "\n"
    model_path = str(
        Path(__file__).parents[1]
    ) + '/models/' + experiment + '/model_' + model_time + '.pth'
    model_info_path = str(
        Path(__file__).parents[1]
    ) + '/models/' + experiment + '/model_info_' + model_time + '.pth'

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
        model_time = trained_model.splitlines()[0]  # remove "\n"
        model_path = str(
            Path(__file__).parents[1]
        ) + '/models/' + experiment + '/model_' + model_time + '.pth'
        # Predictions
        predicted_labels = predict_subject_task_specific(
            model_path, subject, trial, config)
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
    torch.save(trained_model, save_path + '/model_' + time_stamp + '.pth')
    torch.save(trained_model_info,
               save_path + '/model_info_' + time_stamp + '.pth')
    # Save time also
    with open(save_path + '/time.txt', "a") as f:
        f.write(time_stamp + '\n')

    return None
