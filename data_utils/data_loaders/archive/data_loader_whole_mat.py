import numpy as np

from data_utils.data_loaders.archive.data_loader_whole_base import DataLoaderWholeBase


class DataLoaderWholeMat(DataLoaderWholeBase):
    def __init__(self, class_instance, **kwargs):
        super().__init__(class_instance, **kwargs)

    def set_mask_with_labels(self, mask):
        return mask

    @staticmethod
    def labeled_spectrum_get_from_X_y(X, y):
        healthy_spectrum = X[y == 0]
        ill_spectrum = X[y == 1]
        not_certain_spectrum = X[y == 2]

        return healthy_spectrum, ill_spectrum, not_certain_spectrum
