from typing import List

from tensorflow import keras

from data_utils.data_archive.data_archive import DataArchive


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
        data = self.data_archive.get_datas(data_path=self.batch_paths[idx])

        if self.with_sample_weights and self.weights_name in data.keys():
            return data[self.X_name], data[self.y_name], data[self.weights_name]
        else:
            return data[self.X_name], data[self.y_name]
