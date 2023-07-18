import numpy as np

from data_utils.data_loaders.data_loader import DataLoader


class DataLoaderWhole(DataLoader):
    def __init__(self, dict_names=None):
        super().__init__(dict_names=dict_names)
        self.dict_names.append("size")

    def file_read(self, path):
        def reshape(arr):
            return np.reshape(arr, tuple([arr.shape[0] * arr.shape[1]]) + tuple(arr.shape[2:]))

        print(f'Reading {path}')
        if "MASK_PATH" in self.CONFIG_PATHS.keys():
            mask_path = self.CONFIG_PATHS["MASK_PATH"]
        else:
            mask_path = None
        spectrum, mask = self.file_read_mask_and_spectrum(path, mask_path=mask_path)
        mask = self.set_mask_with_label(mask)

        spectrum = self.smooth(spectrum)

        if self.CONFIG_DATALOADER["3D"]:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        size = spectrum.shape[:2]
        X = reshape(spectrum)
        y = reshape(mask)
        indexes_in_datacube = list(np.array(DataLoaderWhole.get_all_indexes(mask)).T)
        values = [X, y, indexes_in_datacube, size]
        values = {n: v for n, v in zip(self.dict_names, values)}

        return values

    def set_mask_with_label(self, mask):
        return self.data_reader.set_mask_with_label(mask)

    @staticmethod
    def get_all_indexes(mask):
        return np.where(np.ones(mask.shape[:2]).astype(bool))
