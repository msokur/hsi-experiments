import tensorflow as tf

from data_utils.tfrecord.tfr_parser import tfr_1d_train_parser, tfr_3d_train_parser
from data_utils.tfrecord.tfr_utils import parse_names_to_int, filter_name_idx_and_labels


class TFRDatasets:
    def __init__(self, data_dir: str, batch_size: int, d3: bool, with_sample_weights: bool):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.d3 = d3
        self.with_sample_weights = with_sample_weights
        self.options = self.__get_tf_options()

    def get_datasets(self, ds_paths: list, train_names: list, valid_names: list, labels: list):
        """Loads a parsed training and validation TFRecord datasets.
        
        :param ds_paths: Paths for the Dataset
        :param train_names: List with names for training data
        :param valid_names: list with names for validation data
        :param labels: List with labels to use for training and validation

        :return: A tuple with the parsed training and validation dataset
        """
        train_ints = self.__get_names_int_list(ds_paths=ds_paths, names=train_names)
        valid_ints = self.__get_names_int_list(ds_paths=ds_paths, names=valid_names)
        tf_labels = tf.Variable(initial_value=labels, dtype=tf.int64)

        dataset = tf.data.TFRecordDataset(filenames=ds_paths)

        if self.d3:
            dataset = dataset.map(map_func=tfr_3d_train_parser)
        else:
            dataset = dataset.map(map_func=tfr_1d_train_parser)

        train_dataset = self.__get_dataset(dataset=dataset, names_int=train_ints, labels=tf_labels)
        valid_dataset = self.__get_dataset(dataset=dataset, names_int=valid_ints, labels=tf_labels)

        return train_dataset, valid_dataset

    @staticmethod
    def __get_names_int_list(ds_paths: list, names: list) -> tf.Variable:
        names_int_dict = parse_names_to_int(tfr_files=ds_paths)
        names_int = []
        for name in names:
            if name in names_int_dict:
                names_int += [names_int_dict[name]]

        return tf.Variable(initial_value=names_int, dtype=tf.int64)

    def __get_dataset(self, dataset: tf.data.TFRecordDataset, names_int: tf.Variable, labels: tf.Variable):
        """Load a TFRecord dataset and pares the date.

        :param dataset: TFRDataset with X, y, sample weights and name indexes
        :param names_int: To use name indexes
        :param labels: To use labels

        :return: Parsed TFRecord dataset
        """
        # --- make every sample able to slice over
        dataset = dataset.flat_map(map_func=lambda X, y, sw, pat_idx: tf.data.Dataset.from_tensor_slices(
                tensors=(X, y, sw, pat_idx)))
        # filter the not used names and labels
        dataset = dataset.map(lambda X, y, sw, pat_idx: filter_name_idx_and_labels(X=X, y=y, sw=sw, pat_idx=pat_idx,
                                                                                   use_pat_idx=names_int,
                                                                                   use_labels=labels))
        if self.with_sample_weights:
            dataset = dataset.map(lambda X, y, sw, pat_idx: (X, y))
        else:
            dataset = dataset.map(lambda X, y, sw, pat_idx: (X, y, sw))

        return dataset.batch(batch_size=self.batch_size).with_options(options=self.options)

    @staticmethod
    def __get_tf_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE

        return options
