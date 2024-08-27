import abc
import os
import pickle
from typing import List, Dict, Union

from tqdm import tqdm

import numpy as np

from data_utils.data_storage import DataStorage
from data_utils.parallel_processing import start_pool_processing

from configuration.parameter import (
    DICT_y, DICT_WEIGHT,
)


class Weights:
    def __init__(self, filename: str, data_storage: DataStorage, labels: list = None, label_file=None,
                 y_dict_name: str = None, weight_dict_name: str = None):
        self.filename = filename
        self.data_storage = data_storage
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

    def weights_get_from_file(self, root_path: str) -> np.ndarray:
        weights_path = os.path.join(root_path, self.filename)
        if os.path.isfile(weights_path):
            weights = pickle.load(open(weights_path, 'rb'))
            return weights['weights']
        else:
            raise ValueError(f"No .weights file was found in the directory, check given path!")

    def weights_get_or_save(self, root_path: str) -> np.ndarray:
        weights_path = os.path.join(root_path, self.filename)

        paths = self.data_storage.get_paths(storage_path=root_path)

        quantities = []
        for path in tqdm(paths):
            y = self.__get_y__(path=path)

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

    def weighted_data_save(self, root_path: str, weights: np.ndarray, is_on_cluster=True, use_parallel_processing=True):
        paths = self.data_storage.get_paths(storage_path=root_path)

        path_indexes = [idx for idx in range(len(paths))]

        if use_parallel_processing:
            start_pool_processing(map_func=self._weighted_data_save_task,
                                  parallel_args=[path_indexes, paths],
                                  is_on_cluster=is_on_cluster,
                                  fix_args=[weights],
                                  print_out="Save weighted data")
        else:
            for path_index, path in tqdm(zip(path_indexes, paths)):
                self._weighted_data_save_task(weights=weights,
                                              path_index=path_index,
                                              path=path)

    def _weighted_data_save_task(self, weights: np.ndarray, path_index: int, path: str):
        y = self.__get_y__(path=path)
        weights_ = np.zeros(y.shape)

        for label in np.unique(y):
            weights_[y == label] = weights[path_index, label]

        self.save_data(path=path, weights=weights_)

    def get_class_weights(self, class_data_paths: List[str]) -> Dict[Union[int, str], float]:
        labels = np.array(self.labels)
        sums = np.zeros(labels.shape)

        for p in class_data_paths:
            y = self.data_storage.get_data(data_path=p, data_name=self.y_dict_name)
            for i, l in enumerate(labels):
                sums[i] += np.flatnonzero(y == l).shape[0]

        total = np.sum(sums)
        weights = {}
        for i, l in enumerate(labels):
            with np.errstate(divide="ignore", invalid="ignore"):
                weights[l] = (1 / sums[i]) * total / len(labels)
            if weights[l] == np.inf:
                weights[l] = 0.0

        return weights

    @abc.abstractmethod
    def save_data(self, path: str, weights: np.ndarray):
        self.data_storage.save_data(save_path=path, data_name=self.weight_dict_name, data=weights)

    def __get_y__(self, path: str) -> np.ndarray:
        return self.data_storage.get_data(data_path=path, data_name=self.y_dict_name)[...]
