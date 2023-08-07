import abc
import os
import pickle
from typing import List

from tqdm import tqdm
from glob import glob

import numpy as np
import zarr

from configuration.parameter import (
    DICT_y, DICT_WEIGHT,
    ZARR_PAT_DATA, PAT_CHUNKS
)


class Weights:
    def __init__(self, filename: str, labels: list = None, label_file: str = None, y_dict_name: str = None,
                 weight_dict_name: str = None):
        self.filename = filename
        if labels is not None:
            self.labels = labels
        elif label_file is not None:
            self.labels = pickle.load(file=open(file=label_file, mode="rb"))
        else:
            raise ValueError("No parameter for labels or a file to load the labels are given."
                             "Please give a List with labels to the parameter 'labels' or a absolute path to the "
                             "parameter 'label_file'.")
        if y_dict_name is None:
            self.y_dict_name = DICT_y
        else:
            self.y_dict_name = y_dict_name
        if weight_dict_name is None:
            self.weight_dict_name = DICT_WEIGHT
        else:
            self.weight_dict_name = weight_dict_name

        self.data = None

    def weights_get_from_file(self, root_path: str) -> np.ndarray:
        weights_path = os.path.join(root_path, self.filename)
        if os.path.isfile(weights_path):
            weights = pickle.load(open(weights_path, 'rb'))
            return weights['weights']
        else:
            raise ValueError(f"No .weights file was found in the directory, check given path!")

    def weights_get_or_save(self, root_path: str) -> np.ndarray:
        weights_path = os.path.join(root_path, self.filename)

        paths = self.get_paths(root_path=root_path)

        quantities = []
        for path in tqdm(paths):
            y = self.get_y(path=path)

            quantity = []
            for y_u in self.labels:
                quantity.append(y[y == y_u].shape[0])

            quantities.append(quantity)

        quantities = np.array(quantities)

        sum_ = np.sum(quantities[:, self.labels])
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = sum_ / quantities

        weights[np.isinf(weights)] = 0

        data = {
            'weights': weights,
            'sum': sum_,
            'quantities': quantities
        }

        with open(weights_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return weights

    def weighted_data_save(self, root_path: str, weights: np.ndarray):
        paths = self.get_paths(root_path=root_path)
        for i, path in tqdm(enumerate(paths)):
            y = self.get_y(path=path)
            weights_ = np.zeros(y.shape)

            for j in np.unique(y):
                weights_[y == j] = weights[i, j]

            self.save_data(path=path, weights=weights_)

    @staticmethod
    @abc.abstractmethod
    def get_paths(root_path: str) -> List[str]:
        pass

    @abc.abstractmethod
    def get_y(self, path: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save_data(self, path: str, weights: np.ndarray):
        pass


class WeightsNPZ(Weights):
    @staticmethod
    def get_paths(root_path: str) -> List[str]:
        return glob(os.path.join(root_path, '*.npz'))

    def get_y(self, path: str) -> np.ndarray:
        self.data = np.load(path)
        return self.data[self.y_dict_name]

    def save_data(self, path: str, weights: np.ndarray):
        data_ = {n: a for n, a in self.data.items()}
        data_[self.weight_dict_name] = weights

        np.savez(path, **data_)


class WeightsZARR(Weights):
    @staticmethod
    def get_paths(root_path: str) -> List[str]:
        paths = []
        zarr_path = os.path.join(root_path, ZARR_PAT_DATA)
        data = zarr.open_group(store=zarr_path)
        for group in data.group_keys():
            paths.append(os.path.abspath(zarr_path) + f"/{data[group].path}")

        return paths

    def get_y(self, path: str) -> np.ndarray:
        self.data = zarr.open_group(store=path, mode="a")
        return self.data[self.y_dict_name][...]

    def save_data(self, path: str, weights: np.ndarray):
        self.data.array(name=self.weight_dict_name, data=weights, chunks=PAT_CHUNKS, overwrite=True)
