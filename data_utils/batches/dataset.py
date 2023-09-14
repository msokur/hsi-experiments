from typing import List

from tensorflow import keras

from data_utils.data_archive import DataArchive


class Dataset(keras.utils.Sequence):
    """Generates data for Keras"""
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
            return self.data_archive.get_batch_data(batch_path=self.batch_paths[idx], X=self.X_name, y=self.y_name,
                                                    weights=self.weights_name)
        else:
            return self.data_archive.get_batch_data(batch_path=self.batch_paths[idx], X=self.X_name, y=self.y_name)
