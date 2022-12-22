import os
import numpy as np
import cv2
from data_utils.hypercube_data import Cube_Read


class DatFile:
    def __init__(self, loader_conf: dict):
        self.loader = loader_conf

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for key, value in self.loader["MASK_COLOR"].items():
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                if len(sub_value) < 4:
                    sub_mask |= ((mask[..., 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                 (mask[:, :, 2] == sub_value[2]))
                else:
                    sub_mask |= ((mask[:, :, 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                 (mask[:, :, 2] == sub_value[2]) & (mask[:, :, 3] > sub_value[3]))
            indexes.append(sub_mask)

        return indexes

    def set_mask_with_label(self, mask):
        result_mask = np.zeros(mask.shape[:2]) - 1
        indexes = self.indexes_get_bool_from_mask(mask)
        for sub_mask, key in zip(indexes, self.loader["MASK_COLOR"].keys()):
            result_mask[sub_mask] = key

        return result_mask

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        spectrum = self.spectrum_read_from_dat(path)

        if mask_path is None:
            path_parts = os.path.split(path)
            mask_path = os.path.join(path_parts[0],
                                     path_parts[1].split(self.loader["MASK_DIFF"][0])[0] + self.loader["MASK_DIFF"][1])
        mask = DatFile.mask_read(mask_path)

        return spectrum, mask

    @staticmethod
    def sort(paths):
        return sorted(paths)

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
        mask = np.r_["-1", mask[..., -2::-1], mask[..., -1:]]

        return mask
