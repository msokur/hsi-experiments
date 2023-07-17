import scipy.io
import abc
from glob import glob
import os
import numpy as np

import config
from data_utils.data_loaders.archive.data_loader_base import DataLoader


# for Esophagus

class DataLoaderMat(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def get_number(self, elem):
        return elem.split("EP")[-1].split('.')[0]

    def sort(self, paths):
        def take_only_number(elem):
            return int(self.get_number(elem))

        paths = sorted(paths, key=take_only_number)
        return paths

    def get_paths_and_splits(self, root_path=config.RAW_NPZ_PATH):
        paths = glob(os.path.join(root_path, '*.npz'))
        paths = self.sort(paths)

        splits = np.array_split(range(len(paths)), config.CROSS_VALIDATION_SPLIT)

        return paths, splits

    def get_labels(self):
        return super().get_labels()

    def get_name(self, path):
        return path.split(config.SYSTEM_PATHS_DELIMITER)[-1].split('.')[0]

    def indexes_get_bool_from_mask(self, mask):
        ill_indexes = (mask == 1)
        healthy_indexes = (mask == 2)
        not_certain_indexes = (mask == 3)

        return healthy_indexes, ill_indexes, not_certain_indexes

    def file_read_mask_and_spectrum(self, path):
        data = scipy.io.loadmat(path)
        spectrum, mask = data['cube'], data['gtMap']

        return spectrum, mask

    @staticmethod
    def labeled_spectrum_get_from_X_y(X, y):
        healthy_spectrum = X[y == 0]
        ill_spectrum = X[y == 1]
        not_certain_spectrum = X[y == 2]

        return healthy_spectrum, ill_spectrum, not_certain_spectrum
