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

    return None


def save_trained_pytorch_model(trained_model, trained_model_info, save_path):

    time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    torch.save(trained_model, save_path
               + '/model_' + time_stamp + '.pth')
    torch.save(trained_model_info, save_path
               + '/model_info_' + time_stamp + '.pth')
    # Save time also
    with open(save_path + '/time.txt', "a") as f:
        f.write(time_stamp + '\n')

    return None
