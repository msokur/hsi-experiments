import scipy.io

import config
from data_loader_base import DataLoader


class DataLoaderMat(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_extension(self):
        return config.FILE_EXTENSIONS['_mat']

    def get_labels(self):
        return super().get_labels()

    def get_name(self, path):
        return path.split(config.SYSTEM_PATHS_DELIMITER)[-1].split('.')[0]

    def indexes_get_bool_from_mask(self, mask):
        healthy_indexes = (mask == 2) | (mask == 3)
        ill_indexes = (mask == 1)
        not_certain_indexes = (mask == 0)

        return healthy_indexes, ill_indexes, not_certain_indexes

    def file_read_mask_and_spectrum(self, path):
        data = scipy.io.loadmat(path)
        spectrum, mask = data['cube'], data['gtMap']

        return spectrum, mask
