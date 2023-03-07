import numpy as np
from utils.common_functions import read_dataframe_file
from easydict import EasyDict


class LinRegDataset():

    def __init__(self, cfg: EasyDict):
        advertising_dataframe = read_dataframe_file(cfg.dataframe_path)
        inputs, targets = np.asarray(advertising_dataframe['inputs']), np.asarray(advertising_dataframe['targets'])
        self.__divide_into_sets(inputs, targets, cfg.train_set_percent, cfg.valid_set_percent)

    def __divide_into_sets(self, inputs: np.ndarray, targets: np.ndarray, train_set_percent: float = 0.8,
                           valid_set_percent: float = 0.1) -> None:
        n_samples = len(inputs)
        shuffle_idx = np.random.permutation(n_samples)
        inputs = inputs[shuffle_idx]
        targets = targets[shuffle_idx]

        n_train = int(train_set_percent * n_samples)
        n_valid = int(valid_set_percent * n_samples)
        n_test = n_samples - n_train - n_valid

        self.inputs_train, self.targets_train = inputs[:n_train], targets[:n_train]
        self.inputs_valid, self.targets_valid = inputs[n_train:n_train + n_valid], targets[n_train:n_train + n_valid]
        self.inputs_test, self.targets_test = inputs[n_train + n_valid:], targets[n_train + n_valid:]

    def __call__(self) -> dict:
        return {'inputs': {'train': self.inputs_train,
                           'valid': self.inputs_valid,
                           'test': self.inputs_test},
                'targets': {'train': self.targets_train,
                            'valid': self.targets_valid,
                            'test': self.targets_test}
                }
