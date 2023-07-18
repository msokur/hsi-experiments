import abc
import numpy as np

from data_utils.data_loaders.archive.data_loader_base import DataLoader
from data_utils.data_loaders.archive.data_loader_colon import DataLoaderColon


class DataLoaderWholeBase(DataLoader):

    def __init__(self, class_instance, **kwargs):
        super().__init__(**kwargs)

        self.class_instance = class_instance
        self.dict_names.append('size')

    def get_labels(self):
        return self.class_instance.get_labels()

    def indexes_get_bool_from_mask(self, mask):
        return self.class_instance.indexes_get_bool_from_mask(mask)

    def file_read_mask_and_spectrum(self, path):
        return self.class_instance.file_read_mask_and_spectrum(path)

    def get_name(self, path):
        return self.class_instance.get_name(path)

    def file_read(self, path):
        def reshape(arr):
            return np.reshape(arr, tuple([arr.shape[0] * arr.shape[1]]) + tuple(arr.shape[2:]))
        print(f'Reading {path}')
        spectrum, mask = self.class_instance.file_read_mask_and_spectrum(path)
        mask = self.set_mask_with_labels(mask)
        
        spectrum = DataLoader.smooth(spectrum)

        #background_mask = DataLoader.background_get_mask(spectrum, mask.shape[:2])  # TODO background in this case is weird, because our goal is to 

        if self._3d:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        #indexes = self.indexes_get_bool_from_mask(mask)
        #indexes = [i * background_mask for i in indexes]

        #spectra = []
        #for idx in indexes:
        #    spectra.append(spectrum[idx])

        #indexes_np = DataLoader.indexes_get_np_from_bool_indexes(*indexes)

        #values = self.X_y_concatenate_from_spectrum(spectra, indexes_np)
        
        print(spectrum.shape, mask.shape, np.unique(mask))
        #print(DataLoaderWholeBase.get_all_indexes(mask)[0].shape)
        size = spectrum.shape[:2]
        X = reshape(spectrum)
        y = reshape(mask)
        indexes_in_datacube = list(np.array(DataLoaderWholeBase.get_all_indexes(mask)).T)
        values = [X, y, indexes_in_datacube, size]
        values = {n: v for n, v in zip(self.dict_names, values)}

        return values
    
    @abc.abstractmethod
    def set_mask_with_labels(self, mask):
        pass
    
    @staticmethod
    def get_all_indexes(mask):
        return np.where(np.ones(mask.shape[:2]).astype(bool))
        