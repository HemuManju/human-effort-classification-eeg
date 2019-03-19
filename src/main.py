from decorators import skip
import yaml
import deepdish as dd
from pathlib import Path
import seaborn as sb
import torch
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import mode
import matplotlib.pyplot as plt
from datetime import datetime
from data.create_eeg_dataset import eeg_dataset
from data.create_robot_dataset import robot_dataset
from data.clean_eeg_dataset import clean_dataset
from data.create_torch_dataset import torch_dataset
from data.create_torch_dataset import balanced_torch_dataset
from features.instability import instability_index
from models.train_model import train
from models.networks import ShallowEEGNet
from models.predict_model import predict_all_task, predict_subject_task_specific
from visualization.visualise import plot_model_accuracy, plot_robot_position
from visualization.visualise import plot_predictions, plot_predictions_with_instability
from decorators import skip_run_code

config = yaml.load(open('config.yml'))


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


with skip_run_code('skip', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[2] / config['raw_eeg_dataset']
    save_dataset(save_path, eeg_dataset, save=True)


with skip_run_code('skip', 'create_robot_dataset') as check, check():
    robot_dataset = robot_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['raw_robot_dataset']
    save_dataset(str(save_path), robot_dataset, save=True)


with skip_run_code('skip', 'clean_eeg_dataset') as check, check():
    clean_dataset = clean_dataset(configp['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(save_path, clean_dataset, save=False)


with skip_run_code('skip', 'torch_dataset') as check, check():
    torch_dataset = create_torch_dataset(subjects, trials, config)
    save_path = str(Path(__file__).parents[1] / config['torch_dataset'])
    save_dataset(save_path, torch_dataset)


with skip_run_code('skip', 'balanced_torch_dataset') as check, check():
    data_path = str(Path(__file__).parents[1] / config['torch_dataset'])
    balanced_dataset = create_balanced_dataset(data_path)
    save_path = str(
        Path(__file__).parents[1] / config['balanced_torch_dataset'])
    save_dataset(save_path, balanced_torch_dataset, save=False)


with skip_run_code('skip', 'traning') as check, check():
    data_path = Path(__file__).parents[1] / \
        'data/processed/balanced_torch_dataset.h5'
    config['data_path'] = str(data_path)
    trained_model, trained_model_info = train(ShallowEEGNet, config)
    save = True
    save_path = str(Path(__file__).parents[1] / 'models')
    if save:
        time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        torch.save(trained_model, save_path +
                   '/model_' + time_stamp + '.pth')
        torch.save(trained_model_info, save_path +
                   '/model_info_' + time_stamp + '.pth')
        # Save time also
        with open(save_path + '/time.txt', "a") as f:
            f.write(time_stamp + '\n')


with skip_run_code('run', 'all_prediction') as check, check():
    read_path = str(Path(__file__).parents[1] / 'models')
    count = 1
    for subject in config['subjects']:
        for trial in config['trials']:
            plt.subplot(len(config['subjects']),len(config['trials']),count)
            with open(read_path + '/time.txt', "r+") as f:
                trained_models = f.readlines()
            # Voting
            all_labels = []
            for trained_model in trained_models:
                trained_model_path = str(
                    Path(__file__).parents[1] / 'models/model_') + trained_model.splitlines()[0] + '.pth'
                # Predictions
                predicted_labels = predict_subject_task_specific(trained_model_path, subject, trial, config)
                # voting system
                all_labels.append(predicted_labels)
            all_labels = np.array(all_labels)
            vote, _ = mode(all_labels, axis=0)
            ins_index = instability_index(subject, trial, config)
            normalised_ins_index = normalize(ins_index[:, np.newaxis], axis=0).ravel()
            plot_predictions_with_instability(subject, trial, config, vote[0], normalised_ins_index)
            # plot_predictions(subject, trial, config, vote[0])
            count = count +1
    plt.show()


with skip_run_code('skip', 'subject_task_specific_prediction') as check, check():
    subject = config['subjects'][0]
    trial = config['trials'][0]
    print(subject, trial)
    read_path = str(Path(__file__).parents[1] / 'models')
    with open(read_path + '/time.txt', "r+") as f:
        trained_models = f.readlines()
    # Voting
    all_labels = []
    for trained_model in trained_models:
        trained_model_path = str(
            Path(__file__).parents[1] / 'models/model_') + trained_model.splitlines()[0] + '.pth'
        # Predictions
        predicted_labels = predict_subject_task_specific(trained_model_path,
                                                        subject, trial, config)
        # voting system
        all_labels.append(predicted_labels)
    all_labels = np.array(all_labels)
    vote, _ = mode(all_labels, axis=0)
    plot_predictions(subject, trial, config, vote[0])
    plt.show()


with skip_run_code('skip', 'prediction_plot') as check, check():
    subject = config['subjects'][0]
    trial = config['trials'][0]
    predictions = predicted_labels[subject][trial]
    plot_predictions(subject, trial, config, predictions)
    plt.show()


with skip_run_code('skip', 'instability_index') as check, check():
    subject = config['subjects'][1]
    trial = config['trials'][0]
    print(subject, trial)
    read_path = str(Path(__file__).parents[1] / 'models')
    with open(read_path + '/time.txt', "r+") as f:
        trained_models = f.readlines()
    # Voting
    all_labels = []
    for trained_model in trained_models:
        trained_model_path = str(
            Path(__file__).parents[1] / 'models/model_') + trained_model.splitlines()[0] + '.pth'
        # Predictions
        predicted_labels = predict_subject_task_specific(trained_model_path,
                                                        subject, trial, config)
        # voting system
        all_labels.append(predicted_labels)
    all_labels = np.array(all_labels)
    vote, _ = mode(all_labels, axis=0)
    ins_index = instability_index(subject, trial, config)
    normalised_ins_index = normalize(ins_index[:, np.newaxis], axis=0).ravel()
    plot_predictions_with_instability(subject, trial, config, vote[0], normalised_ins_index)
    plt.show()
