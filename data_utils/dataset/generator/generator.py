from typing import List, Tuple, Union

import tensorflow as tf
import numpy as np

from data_utils.data_storage import DataStorage


class GeneratorDataset:
    """Dataset for tensorflow"""

    def __init__(self, data_storage: DataStorage, batch_paths: List[str], X_name: str, y_name: str, weights_name: str,
                 with_sample_weights: bool):
        self.data_storage = data_storage
        self.batch_paths = batch_paths
        self.X_name = X_name
        self.y_name = y_name
        self.weights_name = weights_name
        self.with_sample_weights = with_sample_weights

    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, idx):
        batch_data = self.data_storage.get_datas(data_path=self.batch_paths[idx])

        data = (batch_data[self.X_name][:], batch_data[self.y_name][:])

        if self.with_sample_weights and self.weights_name in batch_data:
            data += (batch_data[self.weights_name][:].astype(np.float32),)

        return data

    def __call__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx=idx)

    def get_output_signature(self) -> Union[Tuple[tf.TensorSpec, tf.TensorSpec],
                                            Tuple[tf.TensorSpec, tf.TensorSpec, tf.TensorSpec]]:
        """
        Returns the output signature from the data that the data generator generate for a training.
        Always give the signature from the spectrum and label data. If 'with_sample_weights' True , there is also a
        TensorSpec for the sample weights.
        """
        data = self.data_storage.get_datas(data_path=self.batch_paths[0])
        X, y = data[self.X_name], data[self.y_name]
        output_signature = (
            tf.TensorSpec(shape=((None,) + (X.shape[1:])), dtype=X.dtype, name=self.X_name),
            tf.TensorSpec(shape=(None,) if len(y.shape) == 1 else ((None,) + (y.shape[1:])), dtype=y.dtype,
                          name=self.y_name)
        )

        if self.with_sample_weights:
            output_signature += (tf.TensorSpec(shape=(None,), dtype=tf.float32, name=self.weights_name),)

        return output_signature
