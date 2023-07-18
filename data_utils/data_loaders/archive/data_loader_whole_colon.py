import numpy as np

from data_utils.data_loaders.archive.data_loader_whole_base import DataLoaderWholeBase
from data_utils.data_loaders.archive.data_loader_colon import DataLoaderColon


class DataLoaderWholeColon(DataLoaderWholeBase):
    def __init__(self, **kwargs):
        super().__init__(DataLoaderColon(**kwargs), **kwargs)

    def set_mask_with_labels(self, mask):       
        result_mask = np.zeros(mask.shape[:2]) - 1
        
        result_mask[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)] = 0  #blue = healthy
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)] = 1   #yellow = ill
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 2   #red = not certain
        
        return result_mask

    @staticmethod
    def labeled_spectrum_get_from_X_y(X, y):
        healthy_spectrum = X[y == 0]
        ill_spectrum = X[y == 1]
        not_certain_spectrum = X[y == 2]

        return healthy_spectrum, ill_spectrum, not_certain_spectrum
        