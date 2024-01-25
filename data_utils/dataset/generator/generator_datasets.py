from typing import List, Tuple
from shutil import rmtree

import numpy as np

from data_utils.dataset.dataset_interface import Dataset
from data_utils.dataset.generator import GeneratorDataset
from data_utils.dataset.meta_files import get_shape_from_meta
from data_utils.data_archive import DataArchive
from data_utils.dataset.generator import NameBatchSplit

import tensorflow as tf
import os

from configuration.parameter import (
    TRAIN, VALID, GEN_TYP
)


class GeneratorDatasets(Dataset):
    def __init__(self, batch_size: int, d3: bool, with_sample_weights: bool, data_archive: DataArchive,
                 dict_names: List[str]):
        super().__init__(batch_size, d3, with_sample_weights)
        self.data_archive = data_archive
        self.dict_names = dict_names

    def get_datasets(self, ds_paths: List[str], train_names: List[str], valid_names: List[str], labels: List[int],
                     batch_path: str):
        if not os.path.exists(path=batch_path):
            os.makedirs(name=batch_path)

        self.options = self.__get_ds_options()

        batch_split = NameBatchSplit(data_archive=self.data_archive, batch_size=self.batch_size, use_labels=labels,
                                     dict_names=self.dict_names, with_sample_weights=self.with_sample_weights)
        train_paths, valid_paths = batch_split.split(data_paths=ds_paths, batch_save_path=batch_path,
                                                     except_train_names=train_names, except_valid_names=valid_names,
                                                     train_folder=TRAIN, valid_folder=VALID)

        train_ds = self.__get_dataset__(batch_paths=train_paths, options=self.options)
        valid_ds = self.__get_dataset__(batch_paths=valid_paths, options=self.options)

        return train_ds, valid_ds

    def get_paths(self, root_paths: str) -> List[str]:
        return self.data_archive.get_paths(archive_path=root_paths)

    def get_meta_shape(self, paths: List[str]) -> Tuple[int]:
        return get_shape_from_meta(files=paths, dataset_type=GEN_TYP)

    def get_X(self, path: str, shape: Tuple[int]) -> np.ndarray:
        return self.data_archive.get_data(data_path=path, data_name=self.dict_names[0])

    def delete_batches(self, batch_path: str):
        rmtree(batch_path)

    def __get_dataset__(self, batch_paths: List[str], options: tf.data.Options):
        dataset = GeneratorDataset(data_archive=self.data_archive, batch_paths=batch_paths, X_name=self.dict_names[0],
                                   y_name=self.dict_names[1], weights_name=self.dict_names[5],
                                   with_sample_weights=self.with_sample_weights)
        tf_dataset = tf.data.Dataset.from_generator(generator=dataset, output_signature=dataset.get_output_signature())
        return tf_dataset.with_options(options=options)

    @staticmethod
    def __get_ds_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        return options
