import numpy as np
from sklearn.feature_extraction import image
import abc
import os
from glob import glob
from tqdm import tqdm
import pickle
from scipy.ndimage import gaussian_filter, median_filter

# import configuration.get_config as conf
from data_utils.background_detection import detect_background
import data_utils.border as border
from data_utils.paths.path_splits import get_splits


class DataLoader:
    def __init__(self, config, dict_names=None, _3d=None, _3d_size=None):
        self.config = config
        if dict_names is None:
            dict_names = ['X', 'y', 'indexes_in_datacube']
        self.dict_names = dict_names
        self._3d = _3d
        self._3d_size = _3d_size
        self.fill_3d_params()

    def fill_3d_params(self):
        if self._3d is None:
            self._3d = self.config.CONFIG_DATALOADER['3D']

        if self._3d_size is None:
            self._3d_size = self.config.CONFIG_DATALOADER['3D']

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

    def get_paths_and_splits(self, root_path=None):
        if root_path is None:
            root_path = self.config.CONFIG_PATHS['RAW_NPZ_PATH']

        paths = glob(os.path.join(root_path, '*.npz'))
        paths = sorted(paths)

        splits = get_splits(typ=self.config.CONFIG_DATALOADER["SPLIT_PATHS_BY"], paths=paths,
                            values=self.config.CONFIG_DATALOADER["CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"],
                            delimiter=self.config.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])
        # splits = np.array_split(range(len(paths)), config.CROSS_VALIDATION_SPLIT)

        return paths, splits

    def smooth(self, spectrum):
        smoothing_type = self.config.CONFIG_DATALOADER['SMOOTHING_TYPE']
        smoothing_value = self.config.CONFIG_DATALOADER['SMOOTHING_VALUE']
        if smoothing_type is not None:
            if smoothing_type == 'median_filter':
                spectrum = median_filter(spectrum, size=smoothing_value)
            if smoothing_type == 'gaussian_filter':
                spectrum = gaussian_filter(spectrum, sigma=smoothing_value)
        return spectrum

    def remove_border(self, masks, border_config=None):
        if border_config is None:
            border_config = self.config.CONFIG_DATALOADER['BORDER_CONFIG']

        if border_config['enable']:
            border_masks = []
            for idx, mask in enumerate(masks):
                if idx not in border_config['not_used_labels']:
                    border_method = border_config.BORDERS_CONFIG['methode']
                    if len(border_config['axis']) == 0:
                        border_mask = getattr(border, border_method)(in_arr=masks[idx],
                                                                     d=border_config['depth'])
                    else:
                        border_mask = getattr(border, border_method)(in_arr=masks[idx],
                                                                     d=border_config['depth'],
                                                                     axis=border_config['axis'])
                    border_masks.append(border_mask)
                else:
                    border_masks.append(masks[idx])

            return border_masks

        return masks

    def file_read(self, path):
        print(f'Reading {path}')
        spectrum, mask = self.file_read_mask_and_spectrum(path)

        spectrum = self.smooth(spectrum)

        background_mask = self.background_get_mask(spectrum, mask.shape[:2])
        contamination_mask = self.get_contamination_mask(os.path.split(path)[0], mask.shape[:2])

        if self._3d:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        indexes = self.indexes_get_bool_from_mask(mask)
        indexes = [i * background_mask for i in indexes]
        indexes = [i * contamination_mask for i in indexes]
        border_masks = self.remove_border(indexes)
        indexes = [indexes[i] * border_masks[i] for i in range(len(indexes))]

        spectra = []
        for idx in indexes:
            spectra.append(spectrum[idx])

        indexes_np = DataLoader.indexes_get_np_from_bool_indexes(*indexes)

        values = self.X_y_concatenate_from_spectrum(spectra, indexes_np)
        values = {n: v for n, v in zip(self.dict_names, values)}

        return values

    def files_read_and_save_to_npz(self, root_path, destination_path):
        print('----Saving of .npz archives is started----')

        paths = glob(os.path.join(root_path, "*" + self.config.CONFIG_DATALOADER['FILE_EXTENSION']))

        with open(os.path.join(destination_path, DataLoader.get_labels_filename()), 'wb') as f:
            pickle.dump(self.get_labels(), f, pickle.HIGHEST_PROTOCOL)

        for path in tqdm(paths):
            values = self.file_read(path)
            self.X_y_dict_save_to_npz(path, destination_path, values)

        print('----Saving of .npz archives is over----')

    def patches3d_get_from_spectrum(self, spectrum):
        spectrum_ = np.array([])
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

        tissue_indexes = DataLoader.indexes_get_np_from_bool_indexes(*indexes)

        return tissue_indexes

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

        return X, y, indexes_in_datacube

    @staticmethod
    def get_contamination_mask(path, shape):
        mask = np.full(shape, True)
        contamination_pht = os.path.join(path, DataLoader.get_contamination_filename())
        if os.path.exists(contamination_pht):
            import pandas as pd
            c_in = pd.read_csv(contamination_pht, names=['x-start', 'x-end', 'y-start', 'y-end'], header=0, dtype=int)
            for idx in range(c_in.shape[0]):
                mask[c_in['y-start'][idx]:c_in['y-end'][idx], c_in['x-start'][idx]:c_in['x-end'][idx]] = False
        return mask

    @staticmethod
    def get_labels_filename():
        return "labels.labels"

    @staticmethod
    def get_contamination_filename():
        return 'contamination.csv'

    def get_name_easy(self, path, delimiter=None):
        if delimiter is None:
            delimiter = self.config.CONFIG_PATHS['SYSTEM_PATHS_DELIMITER']
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

        return DataLoader.labeled_spectrum_get_from_X_y(X, y)

    @staticmethod
    @abc.abstractmethod
    def labeled_spectrum_get_from_X_y(X, y):
        pass

    def background_get_mask(self, spectrum, shapes):
        background_mask = np.ones(shapes).astype(np.bool)
        if self.config.CONFIG_DATALOADER['WITH_BACKGROUND_EXTRACTION']:
            background_mask = detect_background(spectrum)
            background_mask = np.reshape(background_mask, shapes)

        return background_mask
