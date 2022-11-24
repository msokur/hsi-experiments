import os
import numpy as np
import cv2
from glob import glob
from data_utils.hypercube_data import Cube_Read


class DatFile:
    def __init__(self, loader_conf: dict):
        self.loader = loader_conf

    def get_paths_and_splits(self, path):
        paths = glob(os.path.join(path, '*.npz'))
        paths = sorted(paths)

        cv_split = int(len(paths) / self.loader["CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"])
        splits = np.array_split(range(len(paths)), cv_split)

        return paths, splits

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for key, value in self.loader["MASK_COLOR"].items():
            if len(value) < 4:
                indexes.append((mask[:, :, 0] == value[0]) & (mask[:, :, 1] == value[1]) & (mask[:, :, 2] == value[2]))
            else:
                indexes.append((mask[:, :, 0] == value[0]) & (mask[:, :, 1] == value[1]) & (mask[:, :, 2] == value[2]) &
                               (mask[:, :, 3] > value[3]))

        return indexes

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        spectrum = self.spectrum_read_from_dat(path)

        if mask_path is None:
            path_parts = os.path.split(path)
            mask_path = os.path.join(path_parts[0],
                                     path_parts[1].split(self.loader["MASK_DIFF"][0])[0] + self.loader["MASK_DIFF"][1])
        mask = DatFile.mask_read(mask_path)

        return spectrum, mask

    def spectrum_read_from_dat(self, dat_path):
        spectrum_data, _ = Cube_Read(dat_path,
                                     wavearea=self.loader["WAVE_AREA"],
                                     Firstnm=self.loader["FIRST_NM"],
                                     Lastnm=self.loader["LAST_NM"]).cube_matrix()

        return spectrum_data

    @staticmethod
    def mask_read(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # read Image with transparency
        # [..., -2::-1] - BGR to RGB, [..., -1:] - only transparency, '-1' - concatenate along last axis
        mask = np.r_['-1', mask[..., -2::-1], mask[..., -1:]]

        return mask
