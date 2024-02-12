import abc
from typing import List, Tuple

import numpy as np

from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK


class Dataset:
    def __init__(self, config):
        """TFRecords dataset from a shuffled dataset

        :param config: Configurations
        """
        self.config = config
        self.batch_size = config.CONFIG_TRAINER[TK.BATCH_SIZE]
        self.d3 = config.CONFIG_DATALOADER[DLK.D3]
        self.with_sample_weights = config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS]
        self.options = self._get_dataset_options()

    @abc.abstractmethod
    def get_datasets(self, dataset_paths: List[str], train_names: List[str], valid_names: List[str], labels: List[int],
                     batch_path: str):
        """Loads a parsed training and validation datasets.

        :param dataset_paths: Paths for the Dataset
        :param train_names: List with names for training data
        :param valid_names: list with names for validation data
        :param labels: List with labels to use for training and validation
        :param batch_path: Root path to save batches

        :return: A tuple with the parsed training and validation dataset
        """
        pass

    @abc.abstractmethod
    def get_dataset_paths(self, root_paths: str) -> List[str]:
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
    def _get_dataset_options():
        pass
