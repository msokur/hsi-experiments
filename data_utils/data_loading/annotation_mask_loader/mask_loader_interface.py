import abc
import os.path
from typing import List

import numpy as np

from configuration.keys import (
    DataLoaderKeys as DLK,
    PathKeys as PK,
)


class AnnotationMaskLoaderInterface:
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def get_mask(self, mask_path: str, shape: tuple) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_boolean_indexes_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def set_mask_with_label(self, mask: np.ndarray) -> np.ndarray:
        pass

    def get_mask_path(self, cube_path: str) -> str:
        path_parts = list(os.path.split(cube_path))

        if DLK.MASK_DIFF in self.config.CONFIG_DATALOADER:
            cube_ending = self.config.CONFIG_DATALOADER[DLK.MASK_DIFF][0]
            mask_ending = self.config.CONFIG_DATALOADER[DLK.MASK_DIFF][1]
            path_parts[1] = path_parts[1].replace(cube_ending, mask_ending)

        if PK.MASK_PATH in self.config.CONFIG_PATHS:
            path_parts[0] = self.config.CONFIG_PATHS[PK.MASK_PATH]

        return str(os.path.join(*path_parts))

    @staticmethod
    @abc.abstractmethod
    def get_extension() -> str:
        pass

    @staticmethod
    def get_coordinates_from_boolean_masks(*args):
        coordinates = []
        for boolean_mask in args:
            coordinates.append(np.where(boolean_mask))

        return coordinates

    @staticmethod
    def _check_shapes(mask_path: str, mask_shape: tuple, expected_shape: tuple):
        if mask_shape != expected_shape:
            raise ValueError(f"The shape of the mask '{mask_path}' is {mask_shape} and the expected shape "
                             f"is {expected_shape}.")

