import numpy as np
from sklearn.feature_extraction import image
import abc
import os
from glob import glob
from tqdm import tqdm
import pickle

import config
from background_detection import detect_background


class DataLoader:
    def __init__(self, dict_names=['X', 'y', 'indexes_in_datacube'], _3d=config._3D, _3d_size=config._3D_SIZE):
        self.dict_names = dict_names
        self._3d = _3d
        self._3d_size = _3d_size
    
    @abc.abstractmethod
    def get_extension(self):
        pass

    @abc.abstractmethod
    def get_labels(self):
        return [0, 1, 2]

    @abc.abstractmethod
    def indexes_get_bool_from_mask(self, mask):
        pass

    @abc.abstractmethod
    def file_read_mask_and_spectrum(self, path):
        pass

    @abc.abstractmethod
    def get_name(self, path):
        pass
    
    def get_paths_and_splits(self, root_path=config.RAW_NPZ_PATH):
        paths = glob(os.path.join(root_path, '*.npz'))
        paths = sorted(paths)

        splits = np.array_split(range(len(paths)), config.CROSS_VALIDATION_SPLIT)
        
        return paths, splits

    def file_read(self, path):
        print(f'Reading {path}')
        spectrum, mask = self.file_read_mask_and_spectrum(path)

        background_mask = DataLoader.background_get_mask(spectrum, mask.shape[:2])

        if self._3d:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        indexes = self.indexes_get_bool_from_mask(mask)
        indexes = [i * background_mask for i in indexes]

        spectra = []
        for idx in indexes:
            spectra.append(spectrum[idx])

        indexes_np = DataLoader.indexes_get_np_from_bool_indexes(*indexes)

        values = self.X_y_concatenate_from_spectrum(spectra, indexes_np)
        values = {n: v for n, v in zip(self.dict_names, values)}

        return values

    def files_read_and_save_to_npz(self, root_path, destination_path):
        print('----Saving of .npz archives is started----')

        paths = glob(os.path.join(root_path, "*" + self.get_extension()))

        with open(os.path.join(destination_path, DataLoader.get_labels_filename()), 'wb') as f:
            pickle.dump(self.get_labels(), f, pickle.HIGHEST_PROTOCOL)

        for path in tqdm(paths):
            values = self.file_read(path)
            self.X_y_dict_save_to_npz(path, destination_path, values)

        print('----Saving of .npz archives is over----')

    def patches3d_get_from_spectrum(self, spectrum):
        size = self._3d_size
        # Better not to use non even sizes
        pad = [int((s - 1) / 2) for s in size]
        if size[0] % 2 == 1 and size[1] % 2 == 1:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        elif size[0] % 2 == 1 and size[1] % 2 == 0:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0]), (pad[1], pad[1] + 1), (0, 0)))
        elif size[0] % 2 == 0 and size[1] % 2 == 1:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0] + 1), (pad[1], pad[1]), (0, 0)))
        elif size[0] % 2 == 0 and size[1] % 2 == 0:
            spectrum_ = np.pad(spectrum, ((pad[0], pad[0] + 1), (pad[1], pad[1] + 1), (0, 0)))

        patches = image.extract_patches_2d(spectrum_, tuple(size))
        patches = np.reshape(patches, (spectrum.shape[0], spectrum.shape[1], size[0], size[1], patches.shape[-1]))

        return patches

    def indexes_get_np_from_mask(self, mask):
        indexes = self.indexes_get_bool_from_mask(mask)

        healthy_indexes, ill_indexes, not_certain_indexes = DataLoader.indexes_get_np_from_bool_indexes(*indexes)

        return healthy_indexes, ill_indexes, not_certain_indexes

    def X_y_dict_save_to_npz(self, path, destination_path, values):
        name = self.get_name(path)
        np.savez(os.path.join(destination_path, name), **{n: a for n, a in values.items()})

    def X_y_concatenate_from_spectrum(self, spectra, indexes, labels=None):
        X, y, indexes_in_datacube = [], [], []
        if labels is None:
            labels = self.get_labels()

        if len(labels) != len(spectra):
            raise ValueError("Error! Labels length doesn't correspond to Spectra length! Check get_labels() and "
                             "indexes_get_bool_from_mask(): whole number of indexes has to be the same as length of "
                             "labels returned from get_labels()")

        if np.unique(labels).shape[0] != len(labels):
            raise ValueError('Error! There are some non unique labels! Check get_labels()')

        for spectrum, label, idx in zip(spectra, labels, indexes):
            X += list(spectrum)
            y += [label] * len(spectrum)
            indexes_in_datacube += list(np.array(idx).T)

        X, y, indexes_in_datacube = np.array(X), np.array(y), np.array(indexes_in_datacube)

        assert X.shape[0] == y.shape[0]
        assert y.shape[0] == indexes_in_datacube.shape[0]

        print(X.shape, y.shape, indexes_in_datacube.shape)

        return X, y, indexes_in_datacube

    @staticmethod
    def get_labels_filename():
        return "labels.labels"

    @staticmethod
    def get_name_easy(path, delimiter=config.SYSTEM_PATHS_DELIMITER):
        return path.split(delimiter)[-1].split(".")[0].split('SpecCube')[0]  # Comments, look at this code:
    # f = 'fff'
    # f.split('v')
    # Out: ['fff'], so if string is split by char that isn't in string than it returns string itself, so an extra check
    # for X does not affect anything, but overlaps one more case

    @staticmethod
    def indexes_get_np_from_bool_indexes(*args):
        indexes_np = []
        for i in args:
            indexes_np.append(np.where(i))

        return indexes_np

    @staticmethod
    def labeled_spectrum_get_from_npz(npz_path):
        data = np.load(npz_path)
        X, y = data['X'], data['y']

        healthy_spectrum, ill_spectrum, not_certain_spectrum = DataLoader.labeled_spectrum_get_from_X_y(X, y)

        return healthy_spectrum, ill_spectrum, not_certain_spectrum

    @staticmethod
    def labeled_spectrum_get_from_X_y(X, y):
        healthy_spectrum = X[y == 0]
        ill_spectrum = X[y == 1]
        not_certain_spectrum = X[y == 2]
        return healthy_spectrum, ill_spectrum, not_certain_spectrum

    @staticmethod
    def background_get_mask(spectrum, shapes):
        background_mask = np.ones(shapes).astype(np.bool)
        if config.WITH_BACKGROUND_EXTRACTION:
            background_mask = detect_background(spectrum)
            background_mask = np.reshape(background_mask, shapes)

        return background_mask
