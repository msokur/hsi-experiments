from typing import List

import numpy as np

import scipy.io as sio

from . import AnnotationMaskLoaderInterface

from configuration.keys import (
    DataLoaderKeys as DLK
)


class MatAnnotationMask(AnnotationMaskLoaderInterface):
    def __init__(self, config):
        super().__init__(config)

    def get_mask(self, mask_path: str, shape: tuple) -> np.ndarray:
        """ Load a mask from a .mat file.

        :param mask_path: The path from the image to load.
        :param shape: The shape of the first two axis of the mask.

        :return: Returns an 3D array with the annotated label for every Pixel.

        :raise ValueError: For file not found or not supported image format.
        """
        data = sio.loadmat(mask_path)
        mask = data[self.config.CONFIG_DATALOADER[DLK.MASK]]

        self._check_shapes(mask_path=mask_path,
                           mask_shape=mask.shape[:2],
                           expected_shape=shape)

        return mask

    def get_boolean_indexes_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """ Create for every classification a boolean array

        :param mask: 2D array with annotation values.

        :return: A list with boolean arrays for every classification

        :raises ValueError: When the given values in "MASK_COLOR" are wrong.
        """
        indexes = []
        for key, value in self.config.CONFIG_DATALOADER[DLK.MASK_COLOR].items():
            if isinstance(value, int):
                raise ValueError(f"Check your configurations for classification {key}! "
                                 f"Surround your value with brackets!")
            elif len(value) == 0:
                raise ValueError(f"Check your configurations for classification {key}! "
                                 f"No annotation value is given!")
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                sub_mask |= (mask == sub_value)
            indexes.append(sub_mask)

        return indexes

    def set_mask_with_label(self, mask: np.ndarray) -> np.ndarray:
        """
        :param mask: Input raw annotation mask

        :return: Array with class number
        """
        result_mask = mask.copy()
        indexes = self.get_boolean_indexes_from_mask(mask=mask)
        for sub_mask, key in zip(indexes, self.config.CONFIG_DATALOADER[DLK.MASK_COLOR].keys()):
            result_mask[sub_mask] = key

        return result_mask

    @staticmethod
    def get_extension() -> str:
        return ".mat"
