import mne
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import signal
from scipy.signal import resample
from datetime import datetime
import yaml
from math import pi
from itertools import product
import pybullet as pb
import pybullet_data
from mne.parallel import parallel_func
import deepdish as dd
from .eeg_utils import get_trial_path, read_eeg_epochs

# Import configuration
path = Path(__file__).parents[1] / 'config.yml'
config = yaml.load(open(path))
epoch_length = config['epoch_length']


def resample_robot_data(x, freq_in, freq_out):
    """Resamples the robot data (force, moment, position, or any general vector x) to desired
    frequency

    Parameters
    ----------
    x : vector
        Description of parameter `x`.
    freq_in : float
        frequency of x signal.
    freq_out : float
        desired frequency of x.

    Returns
    -------
    out : resampled signal with freq_out frequency

    """
    n_samples = round(len(x) * freq_out / freq_in)
    out = resample(x, n_samples)

    return out


def append_xyz(subject, trial):
    """Appends x, y, and z co-ordinates of end effectors to the data file. This function needs to run only once.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine.

    Returns
    -------
    None

    """
    trial_path = get_trial_path(subject, trial)
    joint_angles = np.genfromtxt(trial_path, dtype=float, delimiter=',',
                                 usecols=[1, 2, 3, 4, 5, 6],
                                 skip_header=1).tolist()
    data = np.insert(joint_angles, 0, 0, axis=1)  # Add zero for base of robot
    # Perform forward kinematics to get x, y, and z
    obs = forward_kinematics(np.array(data))
    df = pd.read_csv(trial_path, delimiter=',')
    df[' X'], df[' Y'], df[' Z'] = obs[:, 0], obs[:, 1], obs[:, 2]
    df.to_csv(trial_path, index=False)  # Save the data

    return None


def forward_kinematics(joint_angles):
    """Calculate the poisition of the end effector given joint angles.

    Parameters
    ----------
    joint_angles : array (6 joint angles)
        Joint angles of the data.

    Returns
    -------
    an array
        x, y, and z position of the end effector.

    """
    # Setup the scene for forwad kinematics
    start_pos = [0, 0, 0]
    start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf", start_pos)
    robot_path = path = str(
        Path(__file__).parents[1] / 'power_ball/powerball.urdf')
    robot = pb.loadURDF(robot_path, start_pos,
                        start_orientation, useFixedBase=True)

    obs = []
    pb.setRealTimeSimulation(enableRealTimeSimulation=1)
    for q in joint_angles:
        for _ in range(3):
            pb.setJointMotorControlArray(robot, range(
                7), controlMode=pb.POSITION_CONTROL, targetPositions=q)  # set the joint angles
            pb.stepSimulation()  # Execute the forward kinematics
        obs.append(pb.getLinkState(robot, 6)[0])

    pb.disconnect()  # Terminate the connection

    return np.array(obs)


def get_robot_data(subject, trial):
    """Short summary.

    Parameters
    ----------
    subject         : string of subject ID e.g. 7707
    trial           : trial (str)

    Returns
    ----------
    robot_data : numpy array containing x, y, force_x, force_y, total_force, moment_x, moment_y,
    total_moment, smooth_force. Also start time, end time of trial, as well as duration.

    """
    trial_path = get_trial_path(subject, trial)
    data = np.genfromtxt(trial_path, dtype=float, delimiter=',',
                         usecols=[13, 14, 15, 16, 17, 18, 19, 20],
                         skip_footer=100, skip_header=150).tolist()
    time_data = np.genfromtxt(trial_path, dtype=str, delimiter=',',
                              usecols=0, skip_footer=150,
                              skip_header=100).tolist()
    # Get the sampling frequency
    time = [datetime.strptime(item, '%H:%M:%S:%f') for item in time_data]
    time = np.array(time)  # convert to numpy
    dt = np.diff(time).mean()  # average sampling rate
    freq_in = 1 / dt.total_seconds()
    freq_out = 256.0  # according to eeg sampling rate

    robot_data_resampled = resample_robot_data(data, freq_in, freq_out)

    # Required data
    force_x = robot_data_resampled[:, 0]
    force_y = robot_data_resampled[:, 1]
    total_force = np.linalg.norm(robot_data_resampled[:, 0:2], axis=1)
    moment_x = robot_data_resampled[:, 3]
    moment_y = robot_data_resampled[:, 4]
    total_moment = np.linalg.norm(robot_data_resampled[:, 3:5], axis=1)
    x = robot_data_resampled[:, 6]
    y = robot_data_resampled[:, 7]
    smooth_force = np.mean(total_force) / np.mean(total_moment) * total_moment

    # Stack all the vectors
    robot_data = np.vstack((x, y, force_x, force_y, total_force,
                            moment_x, moment_y, total_moment, smooth_force))
    start_time = time[0]
    end_time = time[-1]
    duration = (time[-1] - time[0]).total_seconds()

    return robot_data, start_time, end_time, duration


def create_robot_epochs(subject, trial):
    """Get the epcohed force data.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine.

    Returns
    -------
    epoch
        epoched robot data.

    """
    data, start_time, end_time, duration = get_robot_data(subject, trial)
    info = mne.create_info(ch_names=['x', 'y', 'force_x', 'force_y',
                                     'total_force', 'moment_x', 'moment_y',
                                     'total_moment', 'smooth_force'],
                           ch_types=['misc'] * data.shape[0],
                           sfreq=256.0)
    raw = mne.io.RawArray(data, info, verbose=False)
    # Additional information
    meas_time = str(start_time) + '..' + str(end_time) + '..' + str(duration)
    raw.info['description'] = meas_time
    raw.info['subject_info'] = subject
    raw.info['experimenter'] = 'hemanth'
    events = mne.make_fixed_length_events(raw, duration=epoch_length)
    epochs = mne.Epochs(raw, events, tmin=0,
                        tmax=epoch_length, verbose=False)

    # Sync with eeg time
    eeg_epochs = read_eeg_epochs(subject, trial)  # eeg file
    drop_id = [id for id, val in enumerate(eeg_epochs.drop_log) if val]
    print(drop_id)
    if len(eeg_epochs.drop_log) != len(epochs.drop_log):
        raise Exception('Two epochs are not of same length!')
    else:
        epochs.drop(drop_id)

    return epochs
