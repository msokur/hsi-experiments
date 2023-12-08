from typing import List

import tensorflow as tf

from data_utils.tfrecord.tfr_parser import tfr_1d_train_parser, tfr_3d_train_parser
from data_utils.tfrecord.tfr_utils import parse_names_to_int


class TFRDatasets:
    def __init__(self, data_dir: str, batch_size: int, d3: bool, with_sample_weights: bool):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.d3 = d3
        self.with_sample_weights = with_sample_weights
        self.options = self.__get_tf_options()

    def get_datasets(self, train_names: list, valid_names: list, labels: list):
        """Loads a parsed training and validation TFRecord datasets.

        :param train_names: List with names for training data
        :param valid_names: list with names for validation data
        :param labels: List with labels to use for training and validation

        :return: A tuple with the parsed training and validation dataset
        """
        train_ints = self.__get_names_int_list(names=train_names)
        valid_ints = self.__get_names_int_list(names=valid_names)

        train_dataset = self.__get_dataset(file=train_names)
        valid_dataset = self.__get_dataset(file=valid_names)

        return train_dataset, valid_dataset

    def __get_names_int_list(self, names: list) -> List[int]:
        names_int_dict = parse_names_to_int(stored_path=self.data_dir)
        names_int = []
        for name in names:
            if name in names_int_dict:
                names_int += names_int_dict[name]

        return names_int

    def __get_dataset(self, file: str):
        """Load a TFRecord dataset and pares the date.

        :param file: Path to TRRecord file

        :return: Parsed TFRecord dataset
        """
        dataset = tf.data.TFRecordDataset(filenames=file)
        dataset = dataset.map(map_func=lambda record: tfr_parser(record=record, X_d3=self.d3,
                                                                 with_sw=self.with_sample_weights))
        # --- make every sample able to slice over
        if self.with_sample_weights:
            dataset = dataset.flat_map(map_func=lambda X, y, sw: tf.data.Dataset.from_tensor_slices(tensors=(X, y, sw)))
        else:
            dataset = dataset.flat_map(map_func=lambda X, y: tf.data.Dataset.from_tensor_slices(tensors=(X, y)))

        return dataset.batch(batch_size=self.batch_size).with_options(options=self.options)

    @staticmethod
    def __get_tf_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE

        return options
