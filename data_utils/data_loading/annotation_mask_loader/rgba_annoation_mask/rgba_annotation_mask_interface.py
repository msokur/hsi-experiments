import abc
import warnings
from typing import List

import numpy as np

from ..mask_loader_interface import AnnotationMaskLoaderInterface

from configuration.keys import (
    DataLoaderKeys as DLK
)


class RGBAAnnotationMaskInterface(AnnotationMaskLoaderInterface):
    def __init__(self, config):
        super().__init__(config)

    @abc.abstractmethod
    def get_mask(self, mask_path: str, shape: tuple) -> np.ndarray:
        pass

    def get_boolean_indexes_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """ Create for every classification a boolean array

        Input is a 3D array with RGB or RGBA code and this function creates for every classification a separate array
        for the annotation by checking the given colors and classes in MASK_COLOR.
        If the color in MASK_COLOR are an RGBA color the result array has only True if the alpha channel
        (last value in array) is highter then the given alpha channel in MASK_COLOR.

        :param mask: 3D array with RGA or RGBA color code.

        :return: A list with boolean arrays for every classification

        :raises ValueError: When the given values in "MASK_COLOR" are wrong.
        :raises Warning: When too many valuse for the RGB/RGBA values

        Example
        -------
        >>> self.config.CONFIG_DATALOADER["MASK_COLOR"]
        dict( 0: [[255, 0, 0]],
        ...   1: [[0, 255, 0]],
        ...   2: [[0, 0, 255]])
        >>> mask
        array( [[[255, 0, 0, 255], [255, 0, 0, 0]],
        ...     [[0, 255, 0, 255], [0, 255, 0, 0]],
        ...     [[0, 0, 255, 255], [0, 0, 255, 0]]])
        >>> result = self.set_mask_with_label(mask)
        list( [[True, True], [False, False], [False, False]],
        ...   [[False, False], [True, True], [False, False]],
        ...   [[False, False], [False, False], [True, True]])

        >>> self.config.CONFIG_DATALOADER["MASK_COLOR"]
        dict( 0: [[255, 0, 0, 200]],
        ...   1: [[0, 255, 0, 200]],
        ...   2: [[0, 0, 255, 200]])
        >>> mask
        array( [[[255, 0, 0, 255], [255, 0, 0, 0]],
        ...     [[0, 255, 0, 255], [0, 255, 0, 0]],
        ...     [[0, 0, 255, 255], [0, 0, 255, 0]]])
        >>> result = self.set_mask_with_label(mask)
        list( [[True, False], [False, False], [False, False]],
        ...   [[False, False], [True, False], [False, False]],
        ...   [[False, False], [False, False], [True, False]])
        """
        indexes = []
        for key, value in self.config.CONFIG_DATALOADER[DLK.MASK_COLOR].items():
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                if isinstance(sub_value, int):
                    raise ValueError(f"Check your configurations in 'MASK_COLOR' for the classification {key}! "
                                     f"Surround your RGB/RGBA value with brackets!")
                elif len(sub_value) < 3:
                    raise ValueError(f"Check your configurations in 'MASK_COLOR' for the classification {key}! "
                                     f"You need a RGB or RGBA value!")

                rgb = np.all(mask[:, :, 0:3] == sub_value[0:3], axis=-1)

                if len(sub_value) >= 4:
                    rgb &= (mask[:, :, 3] > sub_value[3])
                    if len(sub_value) > 4:
                        warnings.warn(f"To many values in 'MASK_COLOR' for the classification {key}! "
                                      f"Only the first four will be used.")
                sub_mask |= rgb
            indexes.append(sub_mask)

        return indexes

    def set_mask_with_label(self, mask: np.ndarray) -> np.ndarray:
        """ Replace  the RGB/RGBA color with keys from MASK_COLOR

        :param mask: The input array with the RGB/RGBA color.

        :return: Return an array with the keys from MASK_COLOR for every RGB/RGBA color. Fields with  an RGB/RGBA
            color not in MASK_COLOR get a -1.

        Example
        -------
        >>> self.config.CONFIG_DATALOADER["MASK_COLOR"]
        dict( 0: [[255, 0, 0]],
        ...   1: [[0, 255, 0]],
        ...   2: [[0, 0, 255], [0, 255, 255]])
        >>> mask
        array( [[[255, 0, 0], [0, 0, 0]],
        ...     [[0, 255, 0], [5, 0, 0]],
        ...     [[0, 0, 255], [0, 255, 255]]])
        >>> result = self.set_mask_with_label(mask)
        array( [[0, -1],
        ...     [1, -1],
        ...     [2, 2]])
        """
        result_mask = np.zeros(mask.shape[:2]) - 1
        indexes = self.get_boolean_indexes_from_mask(mask)
        for sub_mask, key in zip(indexes, self.config.CONFIG_DATALOADER[DLK.MASK_COLOR].keys()):
            result_mask[sub_mask] = key

        return result_mask

    @staticmethod
    @abc.abstractmethod
    def get_extension() -> str:
        pass
