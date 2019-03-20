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
from data.create_eeg_dataset import eeg_dataset
from data.create_robot_dataset import robot_dataset
from data.clean_eeg_dataset import clean_dataset
from data.create_torch_dataset import torch_dataset
from data.create_torch_dataset import balanced_torch_dataset
from features.instability import instability_index
from models.train_model import train
from models.networks import ShallowEEGNet
from models.spatial_model import svm_tangent_space_classifier, svm_tangent_space_prediction
from models.predict_model import predict_all_task, predict_subject_task_specific
from visualization.visualise import plot_model_accuracy, plot_robot_position
from visualization.visualise import plot_predictions, plot_predictions_with_instability
from decorators import skip_run_code
from utils import *

config = yaml.load(open('config.yml'))


with skip_run_code('run', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_dataset(save_path, eeg_dataset, save=True)


with skip_run_code('run', 'clean_eeg_dataset') as check, check():
    clean_dataset = clean_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(save_path, clean_dataset, save=True)


with skip_run_code('run', 'create_robot_dataset') as check, check():
    robot_dataset = robot_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[1] / config['raw_robot_dataset']
    save_dataset(str(save_path), robot_dataset, save=True)


with skip_run_code('run', 'torch_dataset') as check, check():
    torch_dataset = torch_dataset(config['subjects'], config['trials'], config)
    save_path = str(Path(__file__).parents[1] / config['torch_dataset'])
    save_dataset(save_path, torch_dataset, save=True)


with skip_run_code('skip', 'balanced_torch_dataset') as check, check():
    balanced_dataset = create_balanced_dataset(config)
    save_path = Path(__file__).parents[1] / config['balanced_torch_dataset']
    save_dataset(str(save_path), balanced_torch_dataset, save=False)


with skip_run_code('skip', 'training') as check, check():
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


with skip_run_code('skip', 'plot_all_subjects_prediction') as check, check():
    read_path = str(Path(__file__).parents[1] / 'models')
    count = 1
    for subject in config['subjects']:
        for trial in config['trials']:
            plt.subplot(len(config['subjects']), len(config['trials']), count)
            with open(read_path + '/time.txt', "r+") as f:
                trained_models = f.readlines()
            # Voting
            all_labels = []
            for trained_model in trained_models:
                trained_model_path = str(
                    Path(__file__).parents[1] / 'models/model_') + trained_model.splitlines()[0] + '.pth'
                # Predictions
                predicted_labels = predict_subject_task_specific(
                    trained_model_path, subject, trial, config)
                # voting system
                all_labels.append(predicted_labels)
            # Convert to arrays
            all_labels = np.array(all_labels)
            vote, _ = mode(all_labels, axis=0)
            ins_index = instability_index(subject, trial, config)
            # normalised_ins_index = normalize(
            #     ins_index[:, np.newaxis], axis=0).ravel()
            plot_predictions_with_instability(
                subject, trial, config, vote[0], ins_index, details=False)
            # plot_predictions(subject, trial, config, vote[0])
            count = count + 1
    plt.show()


with skip_run_code('skip', 'plot_all_subjects_prediction_svm_classifier') as check, check():
    clf = svm_tangent_space_classifier(config)
    count = 1
    for subject in config['subjects']:
        for trial in config['trials']:
            plt.subplot(len(config['subjects']), len(config['trials']), count)
            predictions = svm_tangent_space_prediction(clf, subject, trial, config)
            ins_index = instability_index(subject, trial, config)
            plot_predictions_with_instability(subject, trial, config, predictions, ins_index, details=False)
            count = count + 1
    plt.show()


with skip_run_code('skip', 'plot_subject_task_specific_prediction') as check, check():
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


with skip_run_code('skip', 'plot_prediction') as check, check():
    subject = config['subjects'][0]
    trial = config['trials'][0]
    predictions = predicted_labels[subject][trial]
    plot_predictions(subject, trial, config, predictions)
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
                Path(__file__).parents[1] / 'models/model_') + trained_model.splitlines()[0] + '.pth'
            # Predictions
            predicted_labels = predict_subject_task_specific(trained_model_path,
                                                             subject, trial, config)
            # voting system
            all_labels.append(predicted_labels)
        all_labels = np.array(all_labels)
        vote, _ = mode(all_labels, axis=0)
        ins_index = instability_index(subject, trial, config)
        plot_predictions_with_instability(
            subject, trial, config, vote[0], ins_index, details=True)
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
        plot_predictions_with_instability(
            subject, trial, config, predictions, ins_index, details=True)
    plt.show()


with skip_run_code('skip', 'all_subjects_spatial_pattern_classification') as check, check():
    clf = svm_tangent_space_classifier(config)
    count = 1
    for subject in config['subjects']:
        for trial in config['trials']:
            plt.subplot(len(config['subjects']), len(config['trials']), count)
            predictions = svm_tangent_space_prediction(clf, subject, trial, config)
            ins_index = instability_index(subject, trial, config)
            plot_predictions_with_instability(
                subject, trial, config, predictions, ins_index, details=False)
            count = count + 1
    plt.show()
