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
        if self.with_sample_weights:
            return self.data_archive.get_batch_datas(batch_path=self.batch_paths[idx], X=self.X_name, y=self.y_name,
                                                     weights=self.weights_name)
        else:
            return self.data_archive.get_batch_datas(batch_path=self.batch_paths[idx], X=self.X_name, y=self.y_name)

    def __call__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx=idx)

    def get_output_signature(self) -> Union[Tuple[tf.TensorSpec, tf.TensorSpec],
                                            Tuple[tf.TensorSpec, tf.TensorSpec, tf.TensorSpec]]:
        X, y, weights = self.data_archive.get_batch_datas(batch_path=self.batch_paths[0], X=self.X_name, y=self.y_name,
                                                          weights=self.weights_name)
        output_signature = (
            tf.TensorSpec(shape=tf.TensorShape(X.shape[1:]), dtype=X.dtype),
            tf.TensorSpec(shape=tf.TensorShape([] if len(y.shape) == 1 else y.shape[1:]), dtype=y.dtype)
        )

        if self.with_sample_weights:
            output_signature += (tf.TensorSpec(shape=tf.TensorShape([]), dtype=weights.dtype),)

        return output_signature
