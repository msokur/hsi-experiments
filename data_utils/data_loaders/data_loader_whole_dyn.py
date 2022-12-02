import numpy as np

from data_utils.data_loaders.data_loader_dyn import DataLoaderDyn


class DataLoaderWhole(DataLoaderDyn):
    def __init__(self):
        super().__init__()
        self.loader["DICT_NAMES"].append("size")

    def file_read(self, path):
        def reshape(arr):
            return np.reshape(arr, tuple([arr.shape[0] * arr.shape[1]]) + tuple(arr.shape[2:]))

        print(f'Reading {path}')
        spectrum, mask = self.file_read_mask_and_spectrum(path)
        mask = self.set_mask_with_label(mask)

        spectrum = self.smooth(spectrum)

        if self.loader["3D"]:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        print(spectrum.shape, mask.shape, np.unique(mask))
        print(DataLoaderWhole.get_all_indexes(mask)[0].shape)
        size = spectrum.shape[:2]
        X = reshape(spectrum)
        y = reshape(mask)
        indexes_in_datacube = list(np.array(DataLoaderWhole.get_all_indexes(mask)).T)
        values = [X, y, indexes_in_datacube, size]
        values = {n: v for n, v in zip(self.loader["DICT_NAMES"], values)}

        return values

    def set_mask_with_label(self, mask):
        if self.loader["FILE_EXTENSIONS"] == ".dat":
            result_mask = np.zeros(mask.shape[:2]) - 1
            for key, value in self.loader["MASK_COLOR"].items():
                if len(value) < 4:
                    result_mask[(mask[:, :, 0] == value[0]) & (mask[:, :, 1] == value[1]) & (mask[:, :, 2] == value[2])] = int(key)
                else:
                    result_mask[(mask[:, :, 0] == value[0]) & (mask[:, :, 1] == value[1]) & (mask[:, :, 2] == value[2]) & (mask[:, :, 3] > value[3])] = int(key)
            return result_mask
        elif self.loader["FILE_EXTENSIONS"] == ".mat":
            return mask
        else:
            raise ValueError(f"For file extension {self.loader['FILE_EXTENSIONS']} is no implementation!")

    @staticmethod
    def get_all_indexes(mask):
        return np.where(np.ones(mask.shape[:2]).astype(bool))
