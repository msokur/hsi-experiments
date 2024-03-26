import abc
from typing import List

import numpy as np
import os
import pickle
from glob import glob
from tqdm import tqdm

import provider

from data_utils.data_loaders.path_splits import get_splits
from data_utils.data_loaders.path_sort import get_sort, folder_sort
from data_utils._3d_patchifier import Patchifier
from data_utils.data_loaders.pixel_masking import BorderMasking, ContaminationMask, Background
from data_utils.data_storage import DataStorage

from configuration.keys import DataLoaderKeys as DLK, PathKeys as PK
from configuration.parameter import (
    DICT_X, DICT_y, DICT_IDX, ORIGINAL_NAME
)


class DataLoader:
    def __init__(self, config, data_storage: DataStorage, dict_names=None):
        self.config = config
        if dict_names is None:
            dict_names = [DICT_X, DICT_y, DICT_IDX]
        if 'background_mask' not in dict_names:
            dict_names.append('background_mask')
        self.data_storage = data_storage
        self.data_reader = provider.get_extension_loader(typ=self.config.CONFIG_DATALOADER[DLK.FILE_EXTENSION],
                                                         config=config)
        self.dict_names = dict_names

    def get_labels(self):
        return self.config.CONFIG_DATALOADER[DLK.LABELS]

    def get_cube_name(self, path: str) -> str:
        return os.path.split(p=path)[-1].split(".")[0].split(self.config.CONFIG_DATALOADER[DLK.NAME_SPLIT])[0]

    def get_name(self, path: str) -> str:
        return self.data_storage.get_name(path=path)

    def get_paths_and_splits(self, root_path=None):
        if root_path is None:
            root_path = self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH]
        paths = self.get_paths(root_path=root_path)
        number = DLK.NUMBER_SORT in self.config.CONFIG_DATALOADER.keys()
        number_sort = self.config.CONFIG_DATALOADER[DLK.NUMBER_SORT] if number else None
        paths = get_sort(paths=paths, number=number, split=number_sort)

        splits = get_splits(typ=self.config.CONFIG_DATALOADER[DLK.SPLIT_PATHS_BY], paths=paths,
                            values=self.config.CONFIG_DATALOADER[DLK.PATIENTS_EXCLUDE_FOR_TEST])

        return paths, splits

    def get_paths(self, root_path) -> List[str]:
        return self.data_storage.get_paths(storage_path=root_path)

    @abc.abstractmethod
    def read_file(self, path):
        print(f'Reading {path}')
        mask_path = self.get_mask_path()
        spectrum, mask = self.read_spectrum_and_mask(path=path, mask_path=mask_path)

        patchifier = Patchifier(self.config)
        use_standard_3D_patchifier, use_onflow_3D_patchifier = patchifier.decide_3D_patchifier(spectrum.shape,
                                                                                               spectrum.dtype)

        spectrum, boolean_masks, background_mask = self.transformations_pipeline(spectrum, mask, path)

        if use_standard_3D_patchifier:
            spectrum = patchifier.get_3D_patches_from_spectrum(spectrum)

        training_instances = self.concatenate_train_instances(spectrum, boolean_masks, background_mask)

        if use_onflow_3D_patchifier:
            training_instances = patchifier.get_3D_patches_onflow(training_instances,
                                                                  spectrum,
                                                                  boolean_masks,
                                                                  background_mask,
                                                                  self.concatenate_train_instances)

        return training_instances

    @staticmethod
    def get_labeled_spectrum_from_boolean_masks(spectrum, boolean_masks):
        spectrum = np.array(spectrum)
        labeled_spectrum = []
        for label_mask in boolean_masks:
            labeled_spectrum.append(spectrum[label_mask])

        return labeled_spectrum

    def transformations_pipeline(self, spectrum, mask, path):
        spectrum = self.smooth(spectrum)
        boolean_masks = self.data_reader.get_boolean_indexes_from_mask(mask)

        transformation_inputs = {
            'config': self.config,
            'path': os.path.split(path)[0],
            'spectrum': spectrum,
            'shape': mask.shape[:2]
        }

        pixel_masking_transformations = [
            Background(**transformation_inputs),
            ContaminationMask(**transformation_inputs),
            BorderMasking(**transformation_inputs)
        ]

        for transformation in pixel_masking_transformations:
            boolean_masks = transformation.process_boolean_masks(boolean_masks)

        background_mask = pixel_masking_transformations[0].background_mask

        return spectrum, boolean_masks, background_mask

    def smooth(self, spectrum):
        if self.config.CONFIG_DATALOADER[DLK.SMOOTHING_TYPE] is not None:
            smoother = provider.get_smoother(typ=self.config.CONFIG_DATALOADER[DLK.SMOOTHING_TYPE],
                                             path="",
                                             size=self.config.CONFIG_DATALOADER[DLK.SMOOTHING_VALUE])
            spectrum = smoother.smooth_func(spectrum)
        return spectrum

    def read_spectrum_and_mask(self, path, mask_path):
        return self.data_reader.file_read_mask_and_spectrum(path=path, mask_path=mask_path)

    def get_mask_path(self):
        if PK.MASK_PATH in self.config.CONFIG_PATHS.keys():
            mask_path = self.config.CONFIG_PATHS[PK.MASK_PATH]
        else:
            mask_path = None

        return mask_path

    def read_files_and_save_to_archive(self, root_path, destination_path):
        print('----Saving of archives is started----')

        paths = self.__get_raw_paths(root_path=root_path)
        with open(os.path.join(destination_path, self.get_labels_filename()), 'wb') as f:
            pickle.dump(self.get_labels(), f, pickle.HIGHEST_PROTOCOL)

        read_and_save_func = self.read_and_save_base

        if DLK.COMBINE_DATA in self.config.CONFIG_DATALOADER:
            if self.config.CONFIG_DATALOADER[DLK.COMBINE_DATA]:
                read_and_save_func = self.read_and_save_folder

        read_and_save_func(paths=paths, destination_path=destination_path)

        print('----Saving of archives is over----')

    def read_and_save_base(self, paths: List[str], destination_path: str):
        for path in tqdm(paths):
            name = self.get_cube_name(path)
            values = self.read_file(path)
            self.save_training_samples_to_archive(destination_path, values, name)

    def read_and_save_folder(self, paths: List[str], destination_path: str):
        names_and_paths = folder_sort(paths=paths)

        for pat_name, paths in tqdm(names_and_paths.items()):
            first = True
            for path in paths:
                name = self.get_cube_name(path=path)
                values = self.read_file(path=path)
                values[ORIGINAL_NAME] = np.array([name] * values[self.dict_names[0]].shape[0])
                if first:
                    self.save_training_samples_to_archive(destination_path=destination_path, values=values,
                                                          name=pat_name)
                    first = False
                else:
                    self.data_storage.append_data(file_path=os.path.join(destination_path, pat_name),
                                                  append_datas=values)

    def get_boolean_masks_from_original_mask(self, mask):
        return self.data_reader.get_boolean_masks_from_original_mask(mask)

    def get_coordinates_from_original_mask(self, mask):
        boolean_masks = self.get_boolean_masks_from_original_mask(mask)

        tissue_indexes = self.get_coordinates_from_boolean_masks(*boolean_masks)

        return tissue_indexes

    def save_training_samples_to_archive(self, destination_path: str, values: dict, name: str) -> None:
        self.data_storage.save_group(save_path=destination_path, group_name=name,
                                     datas=values)

    def concatenate_train_instances(self, spectrum, boolean_masks, background_mask, labels=None):
        labeled_spectrum = self.get_labeled_spectrum_from_boolean_masks(spectrum, boolean_masks)

        coordinates = self.get_coordinates_from_boolean_masks(*boolean_masks)

        X, y, indexes_in_datacube = [], [], []
        if labels is None:
            labels = self.get_labels()

        if len(labels) != len(labeled_spectrum):
            raise ValueError("Error! Labels length doesn't correspond to Spectra length! Check get_labels() and "
                             "get_boolean_masks_from_original_mask(): whole number of indexes has to be the same "
                             "as length of labels returned from get_labels()")

        if np.unique(labels).shape[0] != len(labels):
            raise ValueError("Error! There are some non unique labels! Check get_labels()")

        for spectrum, label, idx in zip(labeled_spectrum, labels, coordinates):
            X += list(spectrum)
            y += [label] * len(spectrum)
            indexes_in_datacube += list(np.array(idx).T)

        X, y, indexes_in_datacube = np.array(X), np.array(y), np.array(indexes_in_datacube)

        assert X.shape[0] == y.shape[0]
        assert y.shape[0] == indexes_in_datacube.shape[0]

        training_instances = X, y, indexes_in_datacube

        training_instances = {n: v for n, v in zip(self.dict_names[:3], training_instances)}
        training_instances["background_mask"] = background_mask

        return training_instances

    def get_labels_filename(self):
        return self.config.CONFIG_DATALOADER[DLK.LABELS_FILENAME]

    @staticmethod
    def get_coordinates_from_boolean_masks(*args):
        coordinates = []
        for boolean_mask in args:
            coordinates.append(np.where(boolean_mask))

        return coordinates

    def get_labeled_spectrum_from_archive(self, path: str) -> dict:
        data = self.data_storage.get_datas(data_path=path)
        X, y = data[self.dict_names[0]], data[self.dict_names[1]]

        return self.get_labeled_spectrum_from_training_samples(X=X, labels=y)

    @staticmethod
    def get_labeled_spectrum_from_training_samples(X: np.ndarray, labels: np.ndarray) -> dict:
        labeled_spectrum = {}
        for label in np.unique(labels):
            labeled_spectrum[label] = X[labels == label]
        return labeled_spectrum

    def __get_raw_paths(self, root_path: str) -> List[str]:
        if DLK.DATA_NAMES_FROM_FILE in self.config.CONFIG_DATALOADER:
            if self.config.CONFIG_DATALOADER[DLK.DATA_NAMES_FROM_FILE] is not None:
                file_name = os.path.join(root_path, self.config.CONFIG_DATALOADER[DLK.DATA_NAMES_FROM_FILE])
                try:
                    import pandas as pd
                    df_names = pd.read_csv(filepath_or_buffer=file_name, header=None)
                    paths = []
                    for name in df_names[0]:
                        paths.append(os.path.join(root_path, name))
                    return paths
                except FileNotFoundError:
                    print(f"File '{file_name}' not found! "
                          f"All files with extension '{self.config.CONFIG_DATALOADER[DLK.FILE_EXTENSION]}' "
                          f"will be used.")

        return glob(os.path.join(root_path, "*" + self.config.CONFIG_DATALOADER[DLK.FILE_EXTENSION]))


if __name__ == "__main__":
    from configuration.configloader_dataloader import read_dataloader_config
    from configuration.configloader_paths import read_path_config
    import configuration.get_config as raw_config

    sys_prefix = r"D:\HTWK\WiSe22\Bachelorarbeit\Programm\hsi-experiments"
    loader_config = os.path.join(sys_prefix, "data_utils", "configuration", "DataLoader.json")
    loader_section = "HNO"
    path_config = os.path.join(sys_prefix, "data_utils", "configuration", "Paths.json")
    system_section = "Win_Benny"
    database_section = "HNO_Database"
    DATALOADER = read_dataloader_config(file=loader_config, section=loader_section)
    PATHS = read_path_config(file=path_config, system_mode=system_section, database=database_section)
    dyn = DataLoader(raw_config)
    x = 1
