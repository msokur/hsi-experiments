from typing import List, Tuple
from shutil import rmtree

import numpy as np

from util.utils import alphanum_key
from ..dataset_interface import Dataset
from ..generator import GeneratorDataset
from ..meta_files import get_shape_from_meta
from .name_batch_split import NameBatchSplit
from data_utils.data_storage import DataStorage

import tensorflow as tf
import os

from configuration.keys import PreprocessorKeys as PPK, CrossValidationKeys as CVK
from configuration.parameter import (
    TRAIN, VALID, GEN_TYP, SKIP_BATCHES, TUNE
)


class GeneratorDatasets(Dataset):
    def __init__(self, config, data_storage: DataStorage):
        super().__init__(config=config)
        self.data_storage = data_storage
        self.dict_names = config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES]

    def get_datasets(self, dataset_paths: List[str], train_names: List[str], valid_names: List[str], labels: List[int],
                     batch_path: str):
        self._error_shuffle_path_size(shuffle_paths=dataset_paths)
        dataset_paths.sort(key=alphanum_key)
        if not os.path.exists(path=batch_path):
            os.makedirs(name=batch_path)

        if self.config.CONFIG_CV[CVK.MODE] == "DEBUG" and len(
                self.data_storage.get_paths(storage_path=os.path.join(batch_path, TRAIN))) > 0:
            train_paths = self.data_storage.get_paths(storage_path=os.path.join(batch_path, TRAIN))
            valid_paths = self.data_storage.get_paths(storage_path=os.path.join(batch_path, VALID))
        else:
            batch_split = NameBatchSplit(data_storage=self.data_storage, batch_size=self.batch_size, use_labels=labels,
                                         dict_names=self.dict_names, with_sample_weights=self.with_sample_weights)
            train_paths, valid_paths = batch_split.split(data_paths=dataset_paths, batch_save_path=batch_path,
                                                         train_names=train_names, valid_names=valid_names,
                                                         train_folder=TRAIN, valid_folder=VALID)

        train_paths.sort(key=alphanum_key)
        valid_paths.sort(key=alphanum_key)

        if self.config.CONFIG_CV[CVK.MODE] == "DEBUG":
            train_paths = train_paths[::SKIP_BATCHES]
            valid_paths = valid_paths[::SKIP_BATCHES]

        self._check_dataset_size(dataset=train_paths, dataset_typ="training")
        train_ds = self.__get_dataset__(batch_paths=train_paths, options=self.options)
        self._check_dataset_size(dataset=valid_paths, dataset_typ="validation")
        valid_ds = self.__get_dataset__(batch_paths=valid_paths, options=self.options)

        return train_ds, valid_ds

    def get_dataset(self, dataset_paths: List[str], names: List[str], labels: List[int], batch_path: str,
                    dataset_type: str):
        self._error_shuffle_path_size(shuffle_paths=dataset_paths)
        dataset_paths.sort(key=alphanum_key)
        if not os.path.exists(path=batch_path):
            os.makedirs(name=batch_path)

        if self.config.CONFIG_CV[CVK.MODE] == "DEBUG" and len(
                self.data_storage.get_paths(storage_path=os.path.join(batch_path, TRAIN))) > 0:
            batch_paths = self.data_storage.get_paths(storage_path=os.path.join(batch_path, dataset_type))
        else:
            batch_split = NameBatchSplit(data_storage=self.data_storage, batch_size=self.batch_size, use_labels=labels,
                                         dict_names=self.dict_names, with_sample_weights=self.with_sample_weights)
            batch_paths, _ = batch_split.split(data_paths=dataset_paths, batch_save_path=batch_path,
                                               train_names=names, valid_names=[],
                                               train_folder=dataset_type, valid_folder="")

        batch_paths.sort(key=alphanum_key)

        if self.config.CONFIG_CV[CVK.MODE] == "DEBUG":
            batch_paths = batch_paths[::SKIP_BATCHES]

        self._check_dataset_size(dataset=batch_paths, dataset_typ=dataset_type)
        dataset = self.__get_dataset__(batch_paths=batch_paths, options=self.options)

        return dataset

    def get_dataset_paths(self, root_paths: str) -> List[str]:
        return self.data_storage.get_paths(storage_path=root_paths)

    def get_meta_shape(self, paths: List[str]) -> Tuple[int]:
        return get_shape_from_meta(files=paths, dataset_type=GEN_TYP)

    def get_X(self, path: str) -> np.ndarray:
        return self.data_storage.get_data(data_path=path, data_name=self.dict_names[0])

    def delete_batches(self, batch_path: str):
        rmtree(batch_path)

    def _check_dataset_size(self, dataset, dataset_typ: str):
        if len(dataset) == 0:
            self._error_dataset_size(dataset_typ=dataset_typ)

    def __get_dataset__(self, batch_paths: List[str], options: tf.data.Options):
        dataset = GeneratorDataset(data_storage=self.data_storage, batch_paths=batch_paths, X_name=self.dict_names[0],
                                   y_name=self.dict_names[1], weights_name=self.dict_names[5],
                                   with_sample_weights=self.with_sample_weights)
        tf_dataset = tf.data.Dataset.from_generator(generator=dataset, output_signature=dataset.get_output_signature())
        return tf_dataset.with_options(options=options)

    @staticmethod
    def _get_dataset_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        return options
