from typing import List, Tuple, Union

import tensorflow as tf

from data_utils.data_archive import DataArchive


class DataGenerator:
    """Dataset for tensorflow"""

    def __init__(self, data_archive: DataArchive, batch_paths: List[str], X_name: str, y_name: str, weights_name: str,
                 with_sample_weights: bool):
        self.data_archive = data_archive
        self.batch_paths = batch_paths
        self.X_name = X_name
        self.y_name = y_name
        self.weights_name = weights_name
        self.with_sample_weights = with_sample_weights

    def __len__(self):
        return len(self.batch_paths)

    def __getitem__(self, idx):
        batch_data = self.data_archive.get_datas(data_path=self.batch_paths[idx])

        if self.with_sample_weights and self.weights_name in batch_data:
            return batch_data[self.X_name], batch_data[self.y_name], batch_data[self.weights_name]

        return batch_data[self.X_name], batch_data[self.y_name]

    def __call__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx=idx)

    def get_output_signature(self) -> Union[Tuple[tf.TensorSpec, tf.TensorSpec],
                                            Tuple[tf.TensorSpec, tf.TensorSpec, tf.TensorSpec]]:
        data = self.data_archive.get_datas(data_path=self.batch_paths[0])
        X, y = data[self.X_name], data[self.y_name]
        output_signature = (
            tf.TensorSpec(shape=((None,) + (X.shape[1:])), dtype=X.dtype, name=self.X_name),
            tf.TensorSpec(shape=(None,) if len(y.shape) == 1 else ((None,) + (y.shape[1:])), dtype=y.dtype,
                          name=self.y_name)
        )
        if self.with_sample_weights:
            weights = data[self.weights_name]
            output_signature += (tf.TensorSpec(shape=(None,), dtype=weights.dtype, name=self.weights_name),)

        return output_signature
