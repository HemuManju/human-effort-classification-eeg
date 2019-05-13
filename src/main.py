import yaml
import deepdish as dd
from pathlib import Path
import seaborn as sb
import torch
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from data.create_eeg_dataset import eeg_dataset
from data.create_robot_dataset import robot_dataset
from data.clean_eeg_dataset import clean_dataset
from data.create_torch_dataset import torch_dataset
from data.create_torch_dataset import balanced_torch_dataset
from features.instability import instability_index
from models.base_model import train
from models.networks import ShallowEEGNet
from models.spatial_model import svm_tangent_space_classifier, svm_tangent_space_prediction
from visualization.visualise import plot_model_accuracy, plot_robot_position
from visualization.visualise import plot_predictions, plot_average_model_accuracy
from utils import *

config = yaml.load(open('config.yml'))

with skip_run_code('skip', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_dataset(str(save_path), eeg_dataset, save=True)

with skip_run_code('skip', 'clean_eeg_dataset') as check, check():
    clean_dataset = clean_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), clean_dataset, save=True)

with skip_run_code('skip', 'create_robot_dataset') as check, check():
    robot_dataset = robot_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['raw_robot_dataset']
    save_dataset(str(save_path), robot_dataset, save=True)

with skip_run_code('skip', 'torch_dataset') as check, check():
    torch_dataset = torch_dataset(config['subjects'], config['trials'], config)
    save_path = str(Path(__file__).parents[1] / config['torch_dataset'])
    save_dataset(save_path, torch_dataset, save=True)

with skip_run_code('skip', 'balanced_torch_dataset') as check, check():
    balanced_dataset = balanced_torch_dataset(config)
    save_path = Path(__file__).parents[1] / config['balanced_torch_dataset']
    save_dataset(str(save_path), balanced_dataset, save=True)

with skip_run_code('skip', 'training') as check, check():
    for _ in range(5):
        trained_model, trained_model_info = train(ShallowEEGNet, config)
        save_path = str(
            Path(__file__).parents[1] / config['trained_model_path'])
        save_trained_pytorch_model(trained_model, trained_model_info,
                                   save_path)

with skip_run_code('skip', 'plot_accuracy') as check, check():
    plot_model_accuracy('experiment_1', config, 1)

with skip_run_code('run', 'plot_average_accuracy') as check, check():
    plot_average_model_accuracy('experiment_1', config)

with skip_run_code('skip', 'plot_all_subjects_prediction') as check, check():
    for i, subject in enumerate(config['subjects']):
        plt.figure(i)
        for j, trial in enumerate(config['trials']):
            plt.subplot(2, 2, j + 1)
            vote = voted_labels('experiment_1', subject, trial, config)
            ins_index = instability_index(subject, trial, config)
            plot_predictions(subject, trial, config, vote, ins_index)
    plt.show()

with skip_run_code('skip', 'plot_svm_prediction') as check, check():
    clf = svm_tangent_space_classifier(config)
    count = 1
    for subject in config['subjects']:
        for trial in config['trials']:
            plt.subplot(len(config['subjects']), len(config['trials']), count)
            predictions = svm_tangent_space_prediction(clf, subject, trial,
                                                       config)
            ins_index = instability_index(subject, trial, config)
            plot_predictions_with_instability(subject,
                                              trial,
                                              config,
                                              predictions,
                                              ins_index,
                                              details=False)
            count = count + 1
    plt.show()

with skip_run_code('skip', 'plot_task_specific_prediction') as check, check():
    subject = config['subjects'][0]
    trial = config['trials'][0]
    vote = voted_labels('experiment_1', subject, trial, config)
    plot_predictions(subject, trial, config, vote, None)
    plt.show()

with skip_run_code('skip', 'plot_subj_specific_prediction') as check, check():
    subject = config['test_subjects'][2]
    trials = config['trials']
    sb.set()
    plt.figure()
    for i, trial in enumerate(config['trials']):
        plt.subplot(len(trials) // 2, len(trials) // 2, i + 1)
        vote = voted_labels('experiment_1', subject, trial, config)
        plot_predictions(subject, trial, config, vote, None)
        plt.title(trial)
    plt.show()

with skip_run_code('skip', 'plot_instability_index') as check, check():
    # name = ['Adaptive damping', 'Low damping', 'High damping']
    for i in range(len(config['trials'])):
        sb.set()
        plt.subplot(1, len(config['trials']), i + 1)
        subject = config['subjects'][6]
        trial = config['trials'][i]
        plt.title(trial)
        print(subject, trial)
        read_path = str(Path(__file__).parents[1] / 'models')
        with open(read_path + '/time.txt', "r+") as f:
            trained_models = f.readlines()
        # Voting
        all_labels = []
        for trained_model in trained_models:
            trained_model_path = str(
                Path(__file__).parents[1] /
                'models/model_') + trained_model.splitlines()[0] + '.pth'
            # Predictions
            predicted_labels = predict_subject_task_specific(
                trained_model_path, subject, trial, config)
            # voting system
            all_labels.append(predicted_labels)
        all_labels = np.array(all_labels)
        vote, _ = mode(all_labels, axis=0)
        ins_index = instability_index(subject, trial, config)
        plot_predictions_with_instability(subject,
                                          trial,
                                          config,
                                          vote[0],
                                          ins_index,
                                          details=True)
    plt.show()

with skip_run_code('skip', 'spatial_pattern_classification') as check, check():

    clf = svm_tangent_space_classifier(config)
    name = ['Adaptive damping', 'Low damping', 'High damping']
    for i in range(3):
        sb.set()
        plt.subplot(1, 3, i + 1)
        subject = config['subjects'][6]
        trial = config['trials'][i]
        plt.title(name[i])
        print(subject, trial)
        predictions = svm_tangent_space_prediction(clf, subject, trial, config)
        ins_index = instability_index(subject, trial, config)
        plot_predictions_with_instability(subject,
                                          trial,
                                          config,
                                          predictions,
                                          ins_index,
                                          details=True)
    plt.show()

with skip_run_code('skip', 'spatial_pattern_classification') as check, check():
    clf = svm_tangent_space_classifier(config)
    count = 1
    for subject in config['subjects']:
        for trial in config['trials']:
            plt.subplot(len(config['subjects']), len(config['trials']), count)
            predictions = svm_tangent_space_prediction(clf, subject, trial,
                                                       config)
            ins_index = instability_index(subject, trial, config)
            plot_predictions(subject, trial, config, predictions, ins_index)
            count = count + 1
    plt.show()
