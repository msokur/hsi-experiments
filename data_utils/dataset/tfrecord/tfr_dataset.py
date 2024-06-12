from typing import List, Tuple

import numpy as np
import tensorflow as tf
import os

from glob import glob
from utils import alphanum_key

from ..dataset_interface import Dataset
from data_utils.dataset.meta_files import get_shape_from_meta
from data_utils.dataset.tfrecord.tfr_utils import get_numpy_X
from data_utils.dataset.tfrecord.tfr_parser import tfr_1d_train_parser, tfr_3d_train_parser
from data_utils.dataset.tfrecord.tfr_utils import filter_name_idx_and_labels
from ..utils import parse_names_to_int

from configuration.keys import CrossValidationKeys as CVK
from configuration.parameter import (
    TFR_FILE_EXTENSION, TFR_TYP
)


class TFRDatasets(Dataset):
    def get_datasets(self, dataset_paths: List[str], train_names: List[str], valid_names: List[str], labels: List[int],
                     batch_path: str):
        dataset_paths.sort(key=alphanum_key)
        if self.config.CONFIG_CV[CVK.MODE] == "DEBUG":
            dataset_paths = [dataset_paths[0]]

        train_ints = self.__get_names_int_list(dataset_paths=dataset_paths, names=train_names)
        valid_ints = self.__get_names_int_list(dataset_paths=dataset_paths, names=valid_names)
        tf_labels = tf.Variable(initial_value=labels, dtype=tf.int64)

        dataset = tf.data.TFRecordDataset(filenames=dataset_paths)

        if self.d3:
            dataset = dataset.map(map_func=tfr_3d_train_parser)
        else:
            dataset = dataset.map(map_func=tfr_1d_train_parser)

        train_dataset = self.__get_dataset(dataset=dataset, names_int=train_ints, labels=tf_labels)
        valid_dataset = self.__get_dataset(dataset=dataset, names_int=valid_ints, labels=tf_labels)

        return train_dataset, valid_dataset

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

    def __get_dataset(self, dataset, names_int: tf.Variable, labels: tf.Variable):
        """Load a TFRecord dataset and pares the date.

        :param dataset: TFRDataset with X, y, sample weights and name indexes
        :param names_int: To use name indexes
        :param labels: To use labels

        :return: Parsed TFRecord dataset
        """
        # filter the not used names and labels
        dataset = dataset.map(lambda X, y, sw, pat_idx: filter_name_idx_and_labels(X=X, y=y, sw=sw, pat_idx=pat_idx,
                                                                                   use_pat_idx=names_int,
                                                                                   use_labels=labels))

        # TODO: Maby refactoring by parsing the dataset to shuffle files
        dataset = dataset.map(map_func=lambda X, y, sw: (X,
                                                         tf.cast(x=tf.expand_dims(input=y, axis=-1), dtype=tf.float32),
                                                         sw))

        if self.with_sample_weights:
            dataset = dataset.flat_map(map_func=lambda X, y, sw: tf.data.Dataset.from_tensor_slices(tensors=(X, y, sw)))
        else:
            dataset = dataset.flat_map(map_func=lambda X, y, sw: tf.data.Dataset.from_tensor_slices(tensors=(X, y)))

        # dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset.batch(batch_size=self.batch_size, drop_remainder=True).with_options(options=self.options)

    @staticmethod
    def _get_dataset_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        return options
