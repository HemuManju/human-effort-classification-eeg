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


def save_dataset(path, dataset, save=False):
    if save:
        dd.io.save(save_path, clean_dataset)


@skip
def processing_raw(config):
    """A Processing pipeline for raw data.

    Parameters
    ----------
    config : yaml file
        A configuration file.

    Returns
    -------
    None

    """

    eeg_dataset = create_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[2] / config['raw_eeg_dataset']
    save_dataset(save_path, eeg_dataset)

    robot_dataset = create_dataset(config['subjects'], config['trials'])
    save_path = Path(__file__).parents[2] / config['raw_robot_dataset']
    save_dataset(save_path, robot_dataset)

@skip
def preprocessing(config):
    """Preprocessing pipeline.

    Parameters
    ----------
    config : yaml file
        A configuration file.

    Returns
    -------
    None

    """

    clean_dataset = clean_dataset(configp['subjects'], config['trials'])
    save_path = Path(__file__).parents[2] / config['clean_eeg_dataset']
    save_dataset(save_path, clean_dataset)


    torch_dataset = create_torch_dataset(subjects, trials, config)
    save_path = str(Path(__file__).parents[2] / config['torch_dataset'])
    save_dataset(save_path, torch_dataset)

    data_path = str(Path(__file__).parents[2] / config['torch_dataset'])
    balanced_dataset = create_balanced_dataset(data_path)
    save_path = str(Path(__file__).parents[2] / config['balanced_torch_dataset'])
    save_dataset(save_path, balanced_torch_dataset)

    return None


def main(config):
    """The main function to execute all other functions.

    """
    processing_raw(config)

    preprocessing(config)


    return None


if __name__ == '__main__':
    # Parameters
    config = yaml.load(open('config.yml'))
    main(config)
