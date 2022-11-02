import numpy as np

from data_utils.data_loaders.data_loader_hno import DataLoaderHNO
from data_utils.data_loaders.data_loader_whole_base import DataLoaderWholeBase


class DataLoaderWholeHNO(DataLoaderWholeBase):
    def __init__(self, **kwargs):
        super().__init__(DataLoaderHNO(**kwargs), **kwargs)

    def set_mask_with_labels(self, mask):
        result_mask = np.zeros(mask.shape[:2]) - 1

        # yellow = nerve
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)] = 0
        # blue = tumour
        result_mask[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)] = 1
        # red = parotis
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 2
        # white = subcutaneous tissue
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)] = 3
        # green = muscle
        result_mask[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)] = 4
        # grey = vein
        result_mask[(mask[:, :, 0] == 128) & (mask[:, :, 1] == 128) & (mask[:, :, 2] == 128)] = 5
        # black = cartilage
        result_mask[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0) & (mask[:, :, 3] >= 125)] = 6
        # light blue = not cartilage
        result_mask[(mask[:, :, 0] == 26) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)] = 7

        return result_mask

    @staticmethod
    def labeled_spectrum_get_from_X_y(X, y):
        nerve_spectrum = X[y == 0]
        tumour_spectrum = X[y == 1]
        parotis_spectrum = X[y == 2]
        subcutaneous_tissue_spectrum = X[y == 3]
        muscle_spectrum = X[y == 4]
        vein_spectrum = X[y == 5]
        cartilage_spectrum = X[y == 6]
        not_certain_spectrum = X[y == 7]
        return nerve_spectrum, tumour_spectrum, parotis_spectrum, subcutaneous_tissue_spectrum, muscle_spectrum, \
               vein_spectrum, cartilage_spectrum, not_certain_spectrum
