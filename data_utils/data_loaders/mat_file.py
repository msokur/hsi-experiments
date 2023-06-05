import numpy as np
import scipy.io as sio


class MatFile:
    def __init__(self, loader_conf: dict):
        self.loader = loader_conf

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for key, value in self.loader["MASK_COLOR"].items:
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                sub_mask |= (mask == sub_value)
            indexes.append(sub_mask)

        return indexes

    def set_mask_with_label(self, mask):
        result_mask = mask.copy()
        indexes = self.indexes_get_bool_from_mask(mask)
        for sub_mask, key in zip(indexes, self.loader["MASK_COLOR"].keys()):
            result_mask[sub_mask] = key

        return result_mask

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        data = sio.loadmat(path)
        spectrum, mask = data[self.loader["SPECTRUM"]], data[self.loader["MASK"]]

        return spectrum, mask
