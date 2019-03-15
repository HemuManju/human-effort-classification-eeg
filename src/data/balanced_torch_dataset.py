import deepdish as dd
import torch
import yaml
from pathlib import Path
import deepdish as dd
import numpy as np


def create_balanced_dataset(data_path):
    """Create balanced pytorch dataset for all subjects.

    Parameters
    ----------
    data_path : str
        Path to pytroch dataset.

    Returns
    -------
    dict
        A balanced dataset.

    """
    balanced_dataset = {}
    data = dd.io.load(data_path)
    features = np.array(data['features'])
    labels = np.array(data['labels'])
    ids = data['data_index']
    # Get each class labels
    class_1_ids = ids[np.argmax(labels, axis=1) == 0]
    class_2_ids = ids[np.argmax(labels, axis=1) == 1]
    class_3_ids = ids[np.argmax(labels, axis=1) == 2]

    # Drop 50% of class 3 labels
    index = np.random.choice(
        class_3_ids.shape[0], int(class_3_ids.shape[0] / 2), replace=False)
    # print(sum(kept) / len(ids))
    class_3_ids = class_3_ids[index]
    # Concatenate all of them
    ids_list = np.hstack((class_1_ids, class_2_ids, class_3_ids))
    balanced_dataset['features'] = features[ids_list, :, :]
    balanced_dataset['labels'] = labels[ids_list, :]
    balanced_dataset['data_index'] = np.arange(ids_list.shape[0])

    return balanced_dataset
