from typing import List, Tuple

import numpy as np
import scipy.io as sio

from configuration.keys import DataLoaderKeys as DLK


class MatFile:
    def __init__(self, config):
        self.CONFIG_DATALOADER = config.CONFIG_DATALOADER

    def indexes_get_bool_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """ Create for every classification a boolean array

        :param mask: 2D array with annotation values.

        :return: A list with boolean arrays for every classification

        :raises ValueError: When the given values in "MASK_COLOR" are wrong.
        """
        indexes = []
        for key, value in self.CONFIG_DATALOADER[DLK.MASK_COLOR].items():
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
        indexes = self.indexes_get_bool_from_mask(mask)
        for sub_mask, key in zip(indexes, self.CONFIG_DATALOADER[DLK.MASK_COLOR].keys()):
            result_mask[sub_mask] = key

        return result_mask

    def file_read_mask_and_spectrum(self, path: str, mask_path=None) -> Tuple[np.ndarray, np.ndarray]:
        """ Load Spectrum and mask

        Load the spectrum and mask from a .mat file.

        For the spectrum you need to specify the name from the dictionary key in the parameter self.loader["SPECTRUM"].

        For the mask you need to specify the name from the dictionary key in the parameter self.loader["MASK"].

        :param path: String with path to .mat file.
        :param mask_path: oNot needed.

        :return: Tuple with a numpy array for the spectrum and a numpy array for the annotation mask.
        """
        data = sio.loadmat(path)
        spectrum, mask = data[self.CONFIG_DATALOADER[DLK.SPECTRUM]], data[self.CONFIG_DATALOADER[DLK.MASK]]

        return spectrum, mask
