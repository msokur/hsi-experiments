import abc
from typing import List

import numpy as np
import pandas as pd
import os
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.feature_extraction import image

import provider
from data_utils.data_archive import DataArchive

from configuration.keys import DataLoaderKeys as DLK, PathKeys as PK
from configuration.parameter import (
    DICT_X, DICT_y, DICT_IDX
)
from data_utils.background_detection import detect_background
from data_utils.data_loaders.path_splits import get_splits
from data_utils.data_loaders.path_sort import get_sort


class DataLoader:
    def __init__(self, data_archive: DataArchive, config_dataloader: dict, config_paths: dict, dict_names=None):
        if dict_names is None:
            dict_names = [DICT_X, DICT_y, DICT_IDX]
        self.data_archive = data_archive
        self.CONFIG_DATALOADER = config_dataloader
        self.CONFIG_PATHS = config_paths
        self.data_reader = provider.get_extension_loader(typ=self.CONFIG_DATALOADER[DLK.FILE_EXTENSION],
                                                         dataloader_config=self.CONFIG_DATALOADER)
        self.dict_names = dict_names

    def get_labels(self):
        return self.CONFIG_DATALOADER[DLK.LABELS]

    def get_cube_name(self, path: str, delimiter=None) -> str:
        if delimiter is None:
            delimiter = self.CONFIG_PATHS[PK.SYS_DELIMITER]
        return path.split(delimiter)[-1].split(".")[0].split(self.CONFIG_DATALOADER[DLK.NAME_SPLIT])[0]

    def get_name(self, path: str) -> str:
        return self.data_archive.get_name(path=path)

    def get_paths(self, root_path) -> List[str]:
        return self.data_archive.get_paths(archive_path=root_path)

    def get_paths_and_splits(self, root_path=None):
        if root_path is None:
            root_path = self.CONFIG_PATHS[PK.RAW_NPZ_PATH]
        paths = self.get_paths(root_path=root_path)
        number = DLK.NUMBER_SORT in self.CONFIG_DATALOADER.keys()
        paths = get_sort(paths=paths, number=number, split=self.CONFIG_DATALOADER[DLK.NUMBER_SORT] if number else None)

        splits = get_splits(typ=self.CONFIG_DATALOADER[DLK.SPLIT_PATHS_BY], paths=paths,
                            values=self.CONFIG_DATALOADER[DLK.PATIENTS_EXCLUDE_FOR_TEST])

        return paths, splits

    def smooth(self, spectrum):
        if self.CONFIG_DATALOADER[DLK.SMOOTHING_TYPE] is not None:
            smoother = provider.get_smoother(typ=self.CONFIG_DATALOADER[DLK.SMOOTHING_TYPE],
                                             path="",
                                             size=self.CONFIG_DATALOADER[DLK.SMOOTHING_VALUE])
            spectrum = smoother.smooth_func(spectrum)
        return spectrum

    def pixel_detection(self, masks, conf=None):
        if conf is None:
            conf = self.CONFIG_DATALOADER[DLK.BORDER_CONFIG]

        if conf[DLK.BC_ENABLE]:
            pixel_detect = provider.get_pixel_detection(conf[DLK.BC_METHODE])
            border_masks = []
            for idx, mask in enumerate(masks):
                if idx not in conf[DLK.BC_NOT_USED_LABELS]:
                    if len(conf[DLK.BC_AXIS]) == 0:
                        border_mask = pixel_detect(in_arr=masks[idx],
                                                   d=conf[DLK.BC_DEPTH])
                    else:
                        border_mask = pixel_detect(in_arr=masks[idx],
                                                   d=conf[DLK.BC_DEPTH],
                                                   axis=conf[DLK.BC_AXIS])
                    border_masks.append(border_mask)
                else:
                    border_masks.append(masks[idx])

            return border_masks

        return masks

    def file_read_mask_and_spectrum(self, path, mask_path):
        return self.data_reader.file_read_mask_and_spectrum(path=path, mask_path=mask_path)

    @abc.abstractmethod
    def file_read(self, path):
        print(f'Reading {path}')
        if PK.MASK_PATH in self.CONFIG_PATHS.keys():
            mask_path = self.CONFIG_PATHS[PK.MASK_PATH]
        else:
            mask_path = None
        spectrum, mask = self.file_read_mask_and_spectrum(path=path, mask_path=mask_path)

        spectrum = self.smooth(spectrum)

        background_mask = self.background_get_mask(spectrum, mask.shape[:2])
        contamination_mask = self.get_contamination_mask(os.path.split(path)[0], mask.shape[:2])

        if self.CONFIG_DATALOADER[DLK.D3]:
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
        values = {n: v for n, v in zip(self.dict_names, values)}

        return values

    def files_read_and_save_to_archive(self, root_path, destination_path):
        print('----Saving of archives is started----')

        paths = glob(os.path.join(root_path, "*" + self.CONFIG_DATALOADER[DLK.FILE_EXTENSION]))
        with open(os.path.join(destination_path, self.get_labels_filename()), 'wb') as f:
            pickle.dump(self.get_labels(), f, pickle.HIGHEST_PROTOCOL)

        for path in tqdm(paths):
            name = self.get_cube_name(path)
            values = self.file_read(path)
            self.X_y_dict_save_to_archive(destination_path, values, name)

        print('----Saving of archives is over----')

    def patches3d_get_from_spectrum(self, spectrum):
        spectrum_ = np.array([])
        size = self.CONFIG_DATALOADER[DLK.D3_SIZE]
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

    def X_y_dict_save_to_archive(self, destination_path: str, values: dict, name: str) -> None:
        self.data_archive.save_group(save_path=destination_path, group_name=name,
                                     datas=values)

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
        return self.CONFIG_DATALOADER[DLK.LABELS_FILENAME]

    def get_contamination_filename(self):
        return self.CONFIG_DATALOADER[DLK.CONTAMINATION_FILENAME]

    def background_get_mask(self, spectrum, shapes):
        background_mask = np.ones(shapes).astype(np.bool)
        if self.CONFIG_DATALOADER[DLK.WITH_BACKGROUND_EXTRACTION]:
            background_mask = detect_background(spectrum)
            background_mask = np.reshape(background_mask, shapes)

        return background_mask

    @staticmethod
    def indexes_get_np_from_bool_indexes(*args):
        indexes_np = []
        for i in args:
            indexes_np.append(np.where(i))

        return indexes_np

    def labeled_spectrum_get_from_archive(self, path: str) -> dict:
        data = self.data_archive.get_datas(data_path=path)
        X, y = data[self.dict_names[0]], data[self.dict_names[1]]

        return self.labeled_spectrum_get_from_X_y(X=X, y=y)

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
    # dyn = DataLoader()
    x = 1
