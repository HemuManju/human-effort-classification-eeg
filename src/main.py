from decorators import skip
import yaml
import deepdish as dd
from pathlib import Path
import seaborn as sb
from visualization.visualise import plot_model_accuracy, plot_robot_position
from models.predict_model import predict
from data.create_eeg_dataset import create_dataset

def get_model_path():
    path = str(Path(__file__).parents[1] / 'models')
    with open(path + '/time.txt', "r+") as f:
        time_stamp = f.readlines()
    time_stamp = time_stamp[-1][0:-1]
    model_path = path + '/model_' + time_stamp + '.pth'
    model_info_path = path + '/model_info_' + time_stamp + '.pth'

    return model_path, model_info_path


def main():


if __name__ == '__main__':
    # Parameters
    config = yaml.load(open('config.yml'))

    model_path, _ = get_model_path()

    # eeg_dataset = create_dataset(config['subjects'], config['trials'])
    # save = True  # Save the file
    # if save:
    #     save_path = Path(__file__).parents[2] / config['raw_eeg_dataset ']
    #     dd.io.save(save_path, eeg_dataset)
    #
    # clean_dataset = clean_dataset(configp['subjects'], config['trials'])
    # save = True  # Save the file
    # if save:
    #     save_path = Path(__file__).parents[2] / config['clean_eeg_dataset']
    #     dd.io.save(save_path, clean_dataset)
    #
    # robot_dataset = create_dataset(config['subjects'], config['trials'])
    # save = True  # Save the file
    # if save:
    #     save_path = Path(__file__).parents[2] / config['raw_robot_dataset']
    #     dd.io.save(save_path, robot_dataset)

    data_path = str(Path(__file__).parents[2] /
                    'data/processed/torch_dataset.h5')
    # Main file
    balanced_dataset = create_balanced_dataset(data_path)
    save = True  # Save the file
    if save:
        save_path = str(Path(__file__).parents[2] / config['balanced_torch_dataset']
        dd.io.save(save_path, balanced_dataset)

    torch_dataset = create_torch_dataset(subjects, trials, config)
    save = True  # Save the file
    if save:
        save_path = str(Path(__file__).parents[2] / config['torch_dataset'])
        dd.io.save(save_path, torch_dataset)
