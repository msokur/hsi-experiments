import abc
import os
import pickle
from typing import List
from glob import glob
import multiprocessing as mp
from functools import partial

import numpy as np

import provider
from data_utils.data_storage import DataStorage
from ..cube_loader import CubeLoaderInterface
from ..annotation_mask_loader import AnnotationMaskLoaderInterface
from data_utils.data_loaders.pixel_masking import BorderMasking, ContaminationMask, Background

from configuration.parameter import (
    DICT_X,
    DICT_y,
    DICT_IDX,
    DICT_BACKGROUND_MASK,
)

from configuration.keys import (
    DataLoaderKeys as DLK
)


class DataLoaderInterface:
    def __init__(self, config, cube_loader: CubeLoaderInterface, mask_loader: AnnotationMaskLoaderInterface,
                 data_storage: DataStorage):
        self.config = config
        self.cube_loader = cube_loader
        self.mask_loader = mask_loader
        self.data_storage = data_storage
        self.dict_names = [DICT_X, DICT_y, DICT_IDX]

    def read_files_and_save_to_archive(self, root_path: str, destination_path: str):
        print('----Saving of archives is started----')

        paths = self._paths_to_load(root_path=root_path)
        with open(os.path.join(destination_path, self._get_labels_filename()), 'wb') as f:
            pickle.dump(self._get_labels(), f, pickle.HIGHEST_PROTOCOL)

        func = partial(self.read_and_save, destination_path, self.config)

        if self.config.CLUSTER:
            # 16 CPUS are set in the job file, so don't change this number
            cpus = 16
        else:
            cpus = mp.cpu_count()
        print(f"----Reading and saving data parallel with {cpus} processors!----")
        with mp.Pool() as pool:
            pool.map(func, paths)

        print('----Saving of archives is over----')

    @classmethod
    @abc.abstractmethod
    def read_and_save(cls, destination_path: str, config, paths: str | List[str]):
        pass

    @classmethod
    def read_data_task(cls, cube_path: str, config):
        cube_loader = provider.get_cube_loader(typ=config.CONFIG_DATALOADER[DLK.FILE_EXTENSION],
                                               config=config)
        mask_loader = provider.get_annotation_mask_loader(typ=config.CONFIG_DATALOADER[DLK.MASK_EXTENSION],
                                                          config=config)

        name = cube_loader.get_cube_name(cube_path=cube_path)

        cube = cube_loader.get_cube(cube_path=cube_path)

        mask_path = mask_loader.get_mask_path(cube_path=cube_path)
        mask = mask_loader.get_mask(mask_path=mask_path,
                                    shape=cube.shape[:2])

        boolean_masks = mask_loader.get_boolean_indexes_from_mask(mask=mask)

        cube, boolean_masks, background_mask = cls.transformations_pipeline(cube=cube,
                                                                            boolean_masks=boolean_masks,
                                                                            cube_path=cube_path,
                                                                            config=config)

        if config.CONFIG_DATALOADER[DLK.D3]:
            from ..patchifier import patching_as_view
            cube = patching_as_view(cube=cube,
                                    patch_size=config.CONFIG_DATALOADER[DLK.D3_SIZE])

        training_instances = cls.concatenate_train_instances(cube, boolean_masks, background_mask, config)

        return name, training_instances

    @staticmethod
    def transformations_pipeline(cube: np.ndarray, boolean_masks: List[np.ndarray], cube_path: str, config):
        if config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_TYPE] is not None:
            smoother = provider.get_smoother(typ=config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_TYPE],
                                             config=config)
            cube = smoother.smooth(spectrum=cube)

        transformation_inputs = {
            "config": config,
            "path": os.path.split(cube_path)[0],
            "spectrum": cube,
            "shape": cube.shape[:2]
        }

        pixel_masking_transformations = [
            Background(**transformation_inputs),
            ContaminationMask(**transformation_inputs),
            BorderMasking(**transformation_inputs)
        ]

        for transformation in pixel_masking_transformations:
            boolean_masks = transformation.process_boolean_masks(boolean_masks)

        background_mask = pixel_masking_transformations[0].background_mask

        return cube, boolean_masks, background_mask

    @classmethod
    def concatenate_train_instances(cls, cube, boolean_masks, background_mask, config, labels=None):
        labeled_spectrum = cls.get_labeled_spectrum_from_boolean_masks(cube, boolean_masks)

        coordinates = AnnotationMaskLoaderInterface.get_coordinates_from_boolean_masks(*boolean_masks)

        X, y, indexes_in_datacube = [], [], []
        if labels is None:
            labels = config.CONFIG_DATALOADER[DLK.LABELS]

        if len(labels) != len(labeled_spectrum):
            raise ValueError("Error! Labels length doesn't correspond to Spectra length! Check get_labels() and "
                             "get_boolean_masks_from_original_mask(): whole number of indexes has to be the same "
                             "as length of labels returned from get_labels()")

        if np.unique(labels).shape[0] != len(labels):
            raise ValueError("Error! There are some non unique labels! Check get_labels()")

        for cube, label, idx in zip(labeled_spectrum, labels, coordinates):
            X += list(cube)
            y += [label] * len(cube)
            indexes_in_datacube += list(np.array(idx).T)

        X, y, indexes_in_datacube = np.array(X), np.array(y), np.array(indexes_in_datacube)

        assert X.shape[0] == y.shape[0]
        assert y.shape[0] == indexes_in_datacube.shape[0]

        training_instances = X, y, indexes_in_datacube

        training_instances = {n: v for n, v in zip([DICT_X, DICT_y, DICT_IDX], training_instances)}
        training_instances[DICT_BACKGROUND_MASK] = background_mask

        return training_instances

    @staticmethod
    def get_labeled_spectrum_from_boolean_masks(spectrum, boolean_masks):
        spectrum = np.array(spectrum)
        labeled_spectrum = []
        for label_mask in boolean_masks:
            labeled_spectrum.append(spectrum[label_mask])

        return labeled_spectrum

    @abc.abstractmethod
    def _paths_to_load(self, root_path: str) -> List[str]:
        pass

    def _get_raw_paths(self, root_path: str) -> List[str]:
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

    def _get_labels(self):
        return self.config.CONFIG_DATALOADER[DLK.LABELS]

    def _get_labels_filename(self):
        return self.config.CONFIG_DATALOADER[DLK.LABELS_FILENAME]

    def get_name(self, path: str) -> str:
        return self.data_storage.get_name(path=path)

    def get_paths(self, root_path) -> List[str]:
        return self.data_storage.get_paths(storage_path=root_path)

    def _save_training_samples_to_archive(self, destination_path: str, values: dict, name: str) -> None:
        self.data_storage.save_group(save_path=destination_path, group_name=name,
                                     datas=values)
