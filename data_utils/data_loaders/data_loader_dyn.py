import abc

import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.feature_extraction import image

import provider_dyn
from configuration.get_config import DATALOADER, PATHS
from data_utils.background_detection import detect_background
from data_utils.data_loaders.path_splits import get_splits


class DataLoaderDyn:
    def __init__(self):
        self.loader = DATALOADER
        self.paths = PATHS
        self.data_reader = provider_dyn.get_extension_loader(typ=self.loader["FILE_EXTENSIONS"],
                                                             loader_conf=self.loader)

    def get_extension(self):
        return self.loader["FILE_EXTENSIONS"]

    def get_labels(self):
        return self.loader["LABELS"]

    def get_name(self, path: str, delimiter=None) -> str:
        if delimiter is None:
            delimiter = self.paths["SYSTEM_PATHS_DELIMITER"]
        return path.split(delimiter)[-1].split(".")[0].split(self.loader["NAME_SPLIT"])[0]

    def get_paths_and_splits(self, root_path=None):
        if root_path is None:
            root_path = self.paths["RAW_NPZ_PATH"]
        paths = self.data_reader.sort(root_path)

        splits = get_splits(typ=self.loader["SPLIT_PATHS_BY"], paths=paths,
                            values=self.loader["CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"],
                            delimiter=self.paths["SYSTEM_PATHS_DELIMITER"])

        return paths, splits

    def smooth(self, spectrum):
        if self.loader["SMOOTHING_TYPE"] is not None:
            smoother = provider_dyn.get_smoother(typ=self.loader["SMOOTHING_TYPE"],
                                                 path="",
                                                 size=self.loader["SMOOTHING_VALUE"])
            spectrum = smoother.smooth_func(spectrum)
        return spectrum

    def pixel_detection(self, masks, conf=None):
        if conf is None:
            conf = self.loader["BORDER_CONFIG"]

        if conf["enable"]:
            pixel_detect = provider_dyn.get_pixel_detection(conf["methode"])
            border_masks = []
            for idx, mask in enumerate(masks):
                if idx not in conf["not_used_labels"]:
                    if len(conf["axis"]) == 0:
                        border_mask = pixel_detect(in_arr=masks[idx],
                                                   d=conf["depth"])
                    else:
                        border_mask = pixel_detect(in_arr=masks[idx],
                                                   d=conf["depth"],
                                                   axis=conf["axis"])
                    border_masks.append(border_mask)
                else:
                    border_masks.append(masks[idx])

            return border_masks

        return masks

    def file_read_mask_and_spectrum(self, path):
        return self.data_reader.file_read_mask_and_spectrum(path)

    @abc.abstractmethod
    def file_read(self, path):
        print(f'Reading {path}')
        spectrum, mask = self.file_read_mask_and_spectrum(path)

        spectrum = self.smooth(spectrum)

        background_mask = self.background_get_mask(spectrum, mask.shape[:2])
        contamination_mask = self.get_contamination_mask(os.path.split(path)[0], mask.shape[:2])

        if self.loader["3D"]:
            spectrum = self.patches3d_get_from_spectrum(spectrum)

        indexes = self.data_reader.indexes_get_bool_from_mask(mask)
        indexes = [i * background_mask for i in indexes]
        indexes = [i * contamination_mask for i in indexes]
        border_masks = self.pixel_detection(indexes)
        indexes = [indexes[i] * border_masks[i] for i in range(len(indexes))]

        spectra = []
        for idx in indexes:
            spectra.append(spectrum[idx])

        indexes_np = self.indexes_get_np_from_bool_indexes(*indexes)

        values = self.X_y_concatenate_from_spectrum(spectra, indexes_np)
        values = {n: v for n, v in zip(self.loader["DICT_NAMES"], values)}

        return values

    def files_read_and_save_to_npz(self, root_path, destination_path):
        print('----Saving of .npz archives is started----')

        paths = glob(os.path.join(root_path, "*" + self.get_extension()))

        with open(os.path.join(destination_path, self.get_labels_filename()), 'wb') as f:
            pickle.dump(self.get_labels(), f, pickle.HIGHEST_PROTOCOL)

        for path in tqdm(paths):
            values = self.file_read(path)
            self.X_y_dict_save_to_npz(path, destination_path, values)

        print('----Saving of .npz archives is over----')

    def patches3d_get_from_spectrum(self, spectrum):
        spectrum_ = np.array([])
        size = self.loader["3D_SIZE"]
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

    def indexes_get_bool_from_mask(self, mask):
        return self.data_reader.indexes_get_bool_from_mask(mask)

    def indexes_get_np_from_mask(self, mask):
        indexes = self.indexes_get_bool_from_mask(mask)

        tissue_indexes = self.indexes_get_np_from_bool_indexes(*indexes)

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
            raise ValueError("Error! There are some non unique labels! Check get_labels()")

        for spectrum, label, idx in zip(spectra, labels, indexes):
            X += list(spectrum)
            y += [label] * len(spectrum)
            indexes_in_datacube += list(np.array(idx).T)

        X, y, indexes_in_datacube = np.array(X), np.array(y), np.array(indexes_in_datacube)

        assert X.shape[0] == y.shape[0]
        assert y.shape[0] == indexes_in_datacube.shape[0]

        return X, y, indexes_in_datacube

    def get_contamination_mask(self, path, shape):
        mask = np.full(shape, True)
        contamination_pht = os.path.join(path, self.get_contamination_filename())
        if os.path.exists(contamination_pht):
            c_in = pd.read_csv(contamination_pht, names=["x-start", "x-end", "y-start", "y-end"], header=0, dtype=int)
            for idx in range(c_in.shape[0]):
                mask[c_in["y-start"][idx]:c_in["y-end"][idx], c_in["x-start"][idx]:c_in["x-end"][idx]] = False
        return mask

    def get_labels_filename(self):
        return self.loader["LABELS_FILENAME"]

    def get_contamination_filename(self):
        return self.loader["CONTAMINATION_FILENAME"]

    def background_get_mask(self, spectrum, shapes):
        background_mask = np.ones(shapes).astype(np.bool)
        if self.loader["WITH_BACKGROUND_EXTRACTION"]:
            background_mask = detect_background(spectrum)
            background_mask = np.reshape(background_mask, shapes)

        return background_mask

    @staticmethod
    def indexes_get_np_from_bool_indexes(*args):
        indexes_np = []
        for i in args:
            indexes_np.append(np.where(i))

        return indexes_np

    @staticmethod
    def labeled_spectrum_get_from_npz(npz_path: str) -> dict:
        data = np.load(npz_path)
        X, y = data['X'], data['y']

        return DataLoaderDyn.labeled_spectrum_get_from_X_y(X, y)

    @staticmethod
    def labeled_spectrum_get_from_X_y(X: np.ndarray, y: np.ndarray) -> dict:
        labeled_spectrum = {}
        for label in np.unique(y):
            labeled_spectrum[label] = X[y == label]
        return labeled_spectrum


if __name__ == "__main__":
    from configuration.configloader_dataloader import read_dataloader_config
    from configuration.configloader_paths import read_path_config

    sys_prefix = r"D:\HTWK\WiSe22\Bachelorarbeit\Programm\hsi-experiments"
    loader_config = os.path.join(sys_prefix, "data_utils", "configuration", "DataLoader.json")
    loader_section = "HNO"
    path_config = os.path.join(sys_prefix, "data_utils", "configuration", "Paths.json")
    system_section = "Win_Benny"
    database_section = "HNO_Database"
    DATALOADER = read_dataloader_config(file=loader_config, section=loader_section)
    PATHS = read_path_config(file=path_config, system_mode=system_section, database=database_section)
    dyn = DataLoaderDyn()
    x = 1
