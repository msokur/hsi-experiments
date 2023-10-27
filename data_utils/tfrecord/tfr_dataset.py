import warnings
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

from data_utils.data_archive import DataArchive
from data_utils.tfrecord.tfr_utils import get_features, tfr_parser_X_y, tfr_parser_X_y_sw
from configuration.parameter import (
    TRAIN, VALID,
)


class TFRDatasets:
    def __init__(self, data_archive: DataArchive, X_name: str, y_name: str, pat_names: str, weights_name: str,
                 with_sample_weights: bool, use_labels: list):
        self.data_archive = data_archive
        self.X_name = X_name
        self.y_name = y_name
        self.pat_names = pat_names
        self.weights_name = weights_name
        self.with_sample_weights = with_sample_weights
        self.use_labels = use_labels
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    def create_train_and_valid_tfrecord_files(self, archive_paths: List[str], out_files_dir: str,
                                              train_names: List[str], valid_names: List[str]) -> Tuple[str, str]:
        train_out_file = self.__get_tfr_file(out_file_dir=out_files_dir, file_name=TRAIN)
        valid_out_file = self.__get_tfr_file(out_file_dir=out_files_dir, file_name=VALID)

        if not os.path.exists(out_files_dir):
            os.makedirs(out_files_dir)
        else:
            self.__delete_tf_file(file_path=train_out_file)
            self.__delete_tf_file(file_path=valid_out_file)

        train_writer = tf.io.TFRecordWriter(path=train_out_file)
        valid_writer = tf.io.TFRecordWriter(path=valid_out_file)

        for p in tqdm(sorted(archive_paths)):
            data = self.data_archive.get_datas(data_path=p)
            X = data[self.X_name][:]
            y = data[self.y_name][:]
            pat_names = data[self.pat_names][:]
            if self.with_sample_weights and self.weights_name in data:
                sample_weights = data[self.weights_name][:]
            else:
                sample_weights = None

            label_indexes = np.isin(y, self.use_labels)

            train_example = self.__get_tf_example(X=X, y=y, pat_names=pat_names, label_indexes=label_indexes,
                                                  except_names=train_names, path=p, sample_weights=sample_weights)
            valid_example = self.__get_tf_example(X=X, y=y, pat_names=pat_names, label_indexes=label_indexes,
                                                  except_names=valid_names, path=p, sample_weights=sample_weights)

            train_writer.write(train_example.SerializeToString())
            valid_writer.write(valid_example.SerializeToString())

        train_writer.close()
        valid_writer.close()

        return train_out_file, valid_out_file

    def get_datasets(self, train_tfr_file: str, valid_tfr_file: str):
        train_dataset = self.__get_dataset(file=train_tfr_file)
        valid_dataset = self.__get_dataset(file=valid_tfr_file)

        return train_dataset, valid_dataset

    def __get_dataset(self, file: str):
        if self.with_sample_weights:
            parse_fn = tfr_parser_X_y_sw
        else:
            parse_fn = tfr_parser_X_y

        dataset = tf.data.TFRecordDataset(filenames=file).with_options(options=self.options)

        return dataset.map(map_func=parse_fn)

    @staticmethod
    def __get_tfr_file(out_file_dir: str, file_name: str) -> str:
        return os.path.join(out_file_dir, file_name + ".tfrecords")

    def __get_tf_example(self, X: np.ndarray, y: np.ndarray, pat_names: np.ndarray, label_indexes: np.ndarray,
                         except_names: List[str], path: str, sample_weights: np.ndarray = None) -> tf.train.Example:
        name_indexes = np.isin(pat_names, except_names)
        indexes = label_indexes & name_indexes
        self.__check_name_in_data(indexes=indexes, data_path=path, names=except_names)

        if sample_weights is not None:
            sample_weights = sample_weights[indexes]

        features = get_features(X=X[indexes], y=y[indexes], sample_weights=sample_weights)

        return tf.train.Example(features=tf.train.Features(feature=features))

    @staticmethod
    def __check_name_in_data(indexes: np.ndarray, data_path: str, names: List[str]):
        if indexes.shape[0] == 0:
            warnings.warn(f"WARING! No data found in {data_path} for the names: {','.join(n for n in names)}.")

    @staticmethod
    def __delete_tf_file(file_path: str):
        """Deletes an existing TF Record File"""
        if os.path.exists(file_path):
            print(f"--- Remove old file: {file_path}")
            os.remove(file_path)
