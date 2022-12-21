import os.path

import numpy as np
import config
from data_loader_base import DataLoader
from data_utils.hypercube_data import Cube_Read


class DataLoaderHNO(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_extension(self):
        return config.FILE_EXTENSIONS['_dat']

    def get_labels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def get_name(self, path):
        return path.split(config.SYSTEM_PATHS_DELIMITER)[-1].split(".")[0].split('_SpecCube')[0]

    def indexes_get_bool_from_mask(self, mask):
        nerve_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)  # yellow
        tumour_indexes = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)  # blue
        parotis_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)  # red
        subcutaneous_tissue_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)  # white
        muscle_indexes = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)  # green
        vein_indexes = (mask[:, :, 0] == 128) & (mask[:, :, 1] == 128) & (mask[:, :, 2] == 128)  # grey
        cartilage_indexes = (mask[:, :, 0] == 12) & (mask[:, :, 1] == 27) & (mask[:, :, 2] == 12)   # black
        not_certain_indexes = (mask[:, :, 0] == 26) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)  # light blue

        return nerve_indexes, tumour_indexes, parotis_indexes, subcutaneous_tissue_indexes, muscle_indexes, \
               vein_indexes, cartilage_indexes, not_certain_indexes

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        spectrum = DataLoaderHNO.spectrum_read_from_dat(path)

        if mask_path is None:
            path_parts = os.path.split(path)
            mask_path = os.path.join(path_parts[0], path_parts[1].replace('_SpecCube.dat', '.png'))
        mask = DataLoaderHNO.mask_read(mask_path)

        return spectrum, mask

    def labeled_spectrum_get_from_dat(self, dat_path, mask_path=None):
        spectrum, mask = self.file_read_mask_and_spectrum(dat_path, mask_path=mask_path)
        nerve_indexes, tumour_indexes, parotis_indexes, subcutaneous_tissue_indexes, muscle_indexes, vein_indexes, \
        cartilage_indexes, not_certain_indexes = self.indexes_get_bool_from_mask(mask)

        return spectrum[nerve_indexes], spectrum[tumour_indexes], spectrum[parotis_indexes], \
               spectrum[subcutaneous_tissue_indexes], spectrum[muscle_indexes], spectrum[vein_indexes], \
               spectrum[cartilage_indexes], spectrum[not_certain_indexes]

    @staticmethod
    def spectrum_read_from_dat(dat_path):
        spectrum_data, _ = Cube_Read(dat_path,
                                     wavearea=config.WAVE_AREA,
                                     Firstnm=config.FIRST_NM,
                                     Lastnm=config.LAST_NM).cube_matrix()
        return spectrum_data

    @staticmethod
    def mask_read(mask_path):
        import cv2
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # read Image with transparency
        # [..., -2::-1] - BGR to RGB, [..., -1:] - only transparency, '-1' - concatenate along last axis
        mask = np.r_['-1', mask[..., -2::-1], mask[..., -1:]]
        return mask

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
