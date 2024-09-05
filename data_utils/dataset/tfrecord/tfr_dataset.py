from typing import List, Tuple

import numpy as np
import tensorflow as tf
import os

from glob import glob
from util.utils import alphanum_key

from ..dataset_interface import Dataset
from data_utils.dataset.meta_files import get_shape_from_meta
from data_utils.dataset.tfrecord.tfr_utils import (
    filter_labels_by_split_factor,
    get_numpy_X,
    filter_name_idx_and_labels,
    skip_every_x_step
)
from data_utils.dataset.tfrecord.tfr_parser import (
    tfr_1d_train_parser,
    tfr_3d_train_parser
)
from ..utils import parse_names_to_int

from configuration.keys import CrossValidationKeys as CVK
from configuration.parameter import (
    TFR_FILE_EXTENSION,
    TFR_TYP,
    SKIP_BATCHES
)


class TFRDatasets(Dataset):
    def get_datasets(self, dataset_paths: List[str], train_names: List[str], valid_names: List[str], labels: List[int],
                     batch_path: str):
        dataset, tf_labels = self._init_dataset(dataset_paths=dataset_paths,
                                                labels=labels)

        train_name_ints = self.__get_names_int_list(dataset_paths=dataset_paths,
                                                    names=train_names)
        valid_name_ints = self.__get_names_int_list(dataset_paths=dataset_paths,
                                                    names=valid_names)

        train_dataset = self.__map_name_dataset(dataset=dataset,
                                                names_int=train_name_ints,
                                                labels=tf_labels)
        valid_dataset = self.__map_name_dataset(dataset=dataset,
                                                names_int=valid_name_ints,
                                                labels=tf_labels)

        self._check_dataset_size(dataset=train_dataset,
                                 dataset_typ="training")
        self._check_dataset_size(dataset=valid_dataset,
                                 dataset_typ="validation")

        return train_dataset, valid_dataset

    def get_datasets_split_factor(self, dataset_paths: List[str], split_factor: float, labels: List[int],
                                  batch_path: str):
        dataset, tf_labels = self._init_dataset(dataset_paths=dataset_paths,
                                                labels=labels)
        tf_split_factor = tf.Variable(split_factor, dtype=tf.float32)

        train_dataset = self.__map_split_factor_dataset(dataset=dataset,
                                                        labels=tf_labels,
                                                        split_factor=tf_split_factor,
                                                        first_part=True)
        valid_dataset = self.__map_split_factor_dataset(dataset=dataset,
                                                        labels=tf_labels,
                                                        split_factor=tf_split_factor,
                                                        first_part=False)

        self._check_dataset_size(dataset=train_dataset,
                                 dataset_typ="training")
        self._check_dataset_size(dataset=valid_dataset,
                                 dataset_typ="validation")

        return train_dataset, valid_dataset

    def _init_dataset(self, dataset_paths: List[str], labels: List[int]):
        self._error_shuffle_path_size(shuffle_paths=dataset_paths)
        dataset_paths.sort(key=alphanum_key)

        tf_labels = tf.Variable(initial_value=labels,
                                dtype=tf.int64)

        dataset = tf.data.TFRecordDataset(filenames=dataset_paths)

        if self.d3:
            dataset = dataset.map(map_func=tfr_3d_train_parser)
        else:
            dataset = dataset.map(map_func=tfr_1d_train_parser)

        return dataset, tf_labels

    def get_dataset_paths(self, root_paths: str) -> List[str]:
        return sorted(glob(os.path.join(root_paths, "*" + TFR_FILE_EXTENSION)))

    def get_meta_shape(self, paths: List[str]) -> Tuple[int]:
        return get_shape_from_meta(files=paths, dataset_type=TFR_TYP)

    def get_X(self, path: str) -> np.ndarray:
        if not path.endswith(TFR_FILE_EXTENSION):
            path += TFR_FILE_EXTENSION
        return get_numpy_X(tfr_path=path)

    def delete_batches(self, batch_path: str):
        pass

    def _check_dataset_size(self, dataset, dataset_typ: str):
        try:
            dataset.as_numpy_iterator().__next__()
        except Exception:
            self._error_dataset_size(dataset_typ=dataset_typ)

    @staticmethod
    def __get_names_int_list(dataset_paths: list, names: list) -> tf.Variable:
        """Parsed names in Dataset files to integer

        :param dataset_paths: List with the paths from the data
        :param names: List with names to use

        :return: A tf Variable with integer
        """
        names_int_dict = parse_names_to_int(files=dataset_paths, meta_type="tfr")
        names_int = []
        for name in names:
            if name in names_int_dict:
                names_int += [names_int_dict[name]]

        return tf.Variable(initial_value=names_int, dtype=tf.int64)

    def __map_name_dataset(self, dataset, names_int: tf.Variable, labels: tf.Variable):
        """Parse a TFRecord dataset with name_ints and labels and batch the dataset

        :param dataset: TFRDataset with X, y, sample weights and name indexes
        :param names_int: To use name indexes
        :param labels: To use labels

        :return: Parsed and batched TFRecord dataset
        """
        # filter the not used names and labels
        dataset = dataset.map(lambda X, y, sw, pat_idx: filter_name_idx_and_labels(X=X, y=y, sw=sw, pat_idx=pat_idx,
                                                                                   use_pat_idx=names_int,
                                                                                   use_labels=labels))

        return self.__batch_dataset(dataset=dataset)

    def __map_split_factor_dataset(self, dataset, labels: tf.Variable, split_factor: tf.Variable,
                                   first_part: bool):
        """Parse a TFRecord dataset and by a split factor

        :param dataset:TFRDataset with X, y, sample weights and name indexes
        :param labels: To use labels
        :param split_factor: To use split factor
        :param first_part: If True, the dataset get the first part of the data, else the last part

        :return: Parsed and batched TFRecord dataset
        """
        tf_first_split = tf.Variable(first_part)
        dataset = dataset.map(lambda X, y, sw, _: filter_labels_by_split_factor(X=X, y=y, sw=sw,
                                                                                use_labels=labels,
                                                                                split_factor=split_factor,
                                                                                first_part=tf_first_split))

        return self.__batch_dataset(dataset=dataset)

    def __batch_dataset(self, dataset):
        """Batch a TFRecord dataset

        :param dataset: TFRecord dataset

        :return: Batched TFRecord dataset
        """
        if self.with_sample_weights:
            dataset = dataset.flat_map(map_func=lambda X, y, sw: tf.data.Dataset.from_tensor_slices(tensors=(X, y, sw)))
        else:
            dataset = dataset.flat_map(map_func=lambda X, y, sw: tf.data.Dataset.from_tensor_slices(tensors=(X, y)))

        # dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True).with_options(options=self.options)

        if self.config.CONFIG_CV[CVK.MODE] == "DEBUG":
            dataset = skip_every_x_step(dataset=dataset,
                                        x_step=SKIP_BATCHES)

        return dataset

    @staticmethod
    def _get_dataset_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        return options
