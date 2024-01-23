import abc
from typing import List, Tuple

import numpy as np


class Dataset:
    def __init__(self, batch_size: int, d3: bool, with_sample_weights: bool):
        """TFRecords dataset from a shuffled dataset

        :param batch_size: Size from batches
        :param d3: True -> data with patches, False -> data without patches
        :param with_sample_weights: True -> use sample weights, False -> don't use sample weights
        """
        self.batch_size = batch_size
        self.d3 = d3
        self.with_sample_weights = with_sample_weights
        self.options = None

    @abc.abstractmethod
    def get_datasets(self, ds_paths: List[str], train_names: List[str], valid_names: List[str], labels: List[int],
                     batch_path: str):
        """Loads a parsed training and validation datasets.

        :param ds_paths: Paths for the Dataset
        :param train_names: List with names for training data
        :param valid_names: list with names for validation data
        :param labels: List with labels to use for training and validation
        :param batch_path: Root path to save batches

        :return: A tuple with the parsed training and validation dataset
        """
        pass

    @abc.abstractmethod
    def get_paths(self, root_paths: str) -> List[str]:
        pass

    @abc.abstractmethod
    def get_meta_shape(self, paths: List[str]) -> Tuple[int]:
        pass

    @abc.abstractmethod
    def get_X(self, path: str, shape: Tuple[int]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def delete_batches(self, batch_path: str):
        pass

    @staticmethod
    @abc.abstractmethod
    def __get_ds_options():
        pass
