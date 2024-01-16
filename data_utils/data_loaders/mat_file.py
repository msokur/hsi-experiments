import numpy as np
import scipy.io as sio


class MatFile:
    def __init__(self, config):
        self.CONFIG_DATALOADER = config.CONFIG_DATALOADER

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for key, value in self.CONFIG_DATALOADER["MASK_COLOR"].items():
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                sub_mask |= (mask == sub_value)
            indexes.append(sub_mask)

        return indexes

    def set_mask_with_label(self, mask):
        result_mask = mask.copy()
        indexes = self.indexes_get_bool_from_mask(mask)
        for sub_mask, key in zip(indexes, self.CONFIG_DATALOADER["MASK_COLOR"].keys()):
            result_mask[sub_mask] = key

        return result_mask

    def get_number(self, elem: str) -> str:
        first_split = elem.split(self.CONFIG_DATALOADER["NUMBER_SPLIT"][0])[-1]
        second_split = first_split.split(".")[0]
        third_split = second_split.split(self.CONFIG_DATALOADER["NUMBER_SPLIT"][1])[0]
        return third_split

    def sort(self, paths):
        def take_only_number(elem):
            return int(self.get_number(elem=elem))

        paths = sorted(paths, key=take_only_number)

        return paths

    def file_read_mask_and_spectrum(self, path):
        data = sio.loadmat(path)
        spectrum, mask = data[self.CONFIG_DATALOADER["SPECTRUM"]], data[self.CONFIG_DATALOADER["MASK"]]

        return spectrum, mask
