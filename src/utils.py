from decorators import skip
import yaml
import deepdish as dd
from pathlib import Path
import seaborn as sb
import torch
import numpy as np


def get_model_path():
    path = str(Path(__file__).parents[1] / 'models')
    with open(path + '/time.txt', "r+") as f:
        time_stamp = f.readlines()
    time_stamp = time_stamp[-1][0:-1]
    model_path = path + '/model_' + time_stamp + '.pth'
    model_info_path = path + '/model_info_' + time_stamp + '.pth'

    return model_path, model_info_path


def save_dataset(path, dataset, save):
    if save:
        dd.io.save(path, dataset)
