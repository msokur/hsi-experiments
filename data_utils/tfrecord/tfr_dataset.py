import tensorflow as tf

from data_utils.tfrecord.tfr_utils import tfr_parser


class TFRDatasets:
    def __init__(self, batch_size: int, d3: bool, with_sample_weights: bool):
        self.batch_size = batch_size
        self.d3 = d3
        self.with_sample_weights = with_sample_weights
        self.options = self.__get_tf_options()

    def get_datasets(self, train_tfr_file: str, valid_tfr_file: str):
        """Loads a parsed training and validation TFRecord datasets.

        :param train_tfr_file: Paths to the training TFRecord file
        :param valid_tfr_file: Paths to the validation TFRecord file

        :return: A tuple with the parsed training and validation dataset
        """
        train_dataset = self.__get_dataset(file=train_tfr_file)
        valid_dataset = self.__get_dataset(file=valid_tfr_file)

        return train_dataset, valid_dataset

    def __get_dataset(self, file: str):
        """Load a TFRecord dataset and pares the date.

        :param file: Path to TRRecord file

        :return: Parsed TFRecord dataset
        """
        dataset = tf.data.TFRecordDataset(filenames=file).with_options(options=self.options)
        dataset = dataset.map(map_func=lambda record: tfr_parser(record=record, X_d3=self.d3,
                                                                 with_sw=self.with_sample_weights))
        # --- make every sample able to slice over
        if self.with_sample_weights:
            dataset = dataset.flat_map(map_func=lambda X, y, sw: tf.data.Dataset.from_tensor_slices(tensors=(X, y, sw)))
        else:
            dataset = dataset.flat_map(map_func=lambda X, y: tf.data.Dataset.from_tensor_slices(tensors=(X, y)))
        return dataset.batch(batch_size=self.batch_size)

    @staticmethod
    def __get_tf_options():
        """Get TF data options"""
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        return options
