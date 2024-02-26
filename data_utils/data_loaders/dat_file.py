import os
from typing import Tuple, List
import numpy as np
import warnings
from data_utils.hypercube_data import HyperCube

from configuration.keys import DataLoaderKeys as DLK


class DatFile:
    def __init__(self, config):
        """ Class to load .dat files and annotations masks

        This class can load a HSI spectrum from a .dat file and annotation masks from .png, .jpg, .jpeg, .jpe or .mk2
        file.

        The input dictionary for config.CONFIG_DATALOADER need following keys and values:

        MASK_COLOR -> dictionary with int keys (start by 0) with RGB and/or RGBA values.

        MASK_COLOR: {0: [[255, 0, 0]], 1: [[0, 255, 0]]}

        TISSUE_LABELS -> dictionary with int keys (start by 0) with name for classifications.

        MASK_COLOR: {0: 'Class0', 1: 'Class1'}

        MASK_DIFF -> list with difference between the file name from the .dat file and the mask file.

        MASK_DIFF: ['.dat', '.png']

        WAVE_AREA -> integer for the length from the HSI-cube.

        FIRST_NM -> integer for the first reflection to read from the HSI-cube.

        LAST_NM -> integer for the last reflection to read from the HSI-cube (not included).

        see also: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Configuration/DataLoader-Configuration

        :param config: configuration
        """
        self.CONFIG_DATALOADER = config.CONFIG_DATALOADER

    def indexes_get_bool_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
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
        >>> self.CONFIG_DATALOADER["MASK_COLOR"]
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

        >>> self.CONFIG_DATALOADER["MASK_COLOR"]
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
        for key, value in self.CONFIG_DATALOADER[DLK.MASK_COLOR].items():
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                if isinstance(sub_value, int):
                    raise ValueError(f"Check your configurations in 'MASK_COLOR' for the classification {key}! "
                                     f"Surround your RGB/RGBA value with brackets!")
                elif len(sub_value) < 3:
                    raise ValueError(f"Check your configurations in 'MASK_COLOR' for the classification {key}! "
                                     f"You need a RGB or RGBA value!")

                rgb = (mask[:, :, 0:3] == sub_value[0:3]).all(axis=-1)

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
        >>> self.CONFIG_DATALOADER["MASK_COLOR"]
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
        indexes = self.indexes_get_bool_from_mask(mask)
        for sub_mask, key in zip(indexes, self.CONFIG_DATALOADER[DLK.MASK_COLOR].keys()):
            result_mask[sub_mask] = key

        return result_mask

    def file_read_mask_and_spectrum(self, path: str, mask_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Load Spectrum and mask

        Load the spectrum from the .dat file and the mask file. If the mask file in a different folder as the .dat file
        you need to give the path to the parameter mask_path.

        :param path: String with path to .dat file.
        :param mask_path: optional - path to mask file.

        :return: Tuple with a numpy array for the spectrum and a numpy array for the annotation mask.
        """
        spectrum = self.spectrum_read_from_dat(path)

        path_parts = os.path.split(path)
        name = path_parts[1].split(self.CONFIG_DATALOADER[DLK.MASK_DIFF][0])[0] \
               + self.CONFIG_DATALOADER[DLK.MASK_DIFF][1]

        if mask_path is None:
            mask_path = os.path.join(path_parts[0], name)
        else:
            mask_path = os.path.join(mask_path, name)

        if mask_path.endswith(".mk2"):
            mask = self.mk2_mask(mask_path, spectrum.shape[0:2])
        else:
            mask = DatFile.mask_read(mask_path)

        return spectrum, mask

    def spectrum_read_from_dat(self, dat_path: str) -> np.ndarray:
        """ Load the HSI-cube

        :param dat_path: path to .dat file

        :return: Numpy array with the spectrum
        """
        spectrum_data = HyperCube(address=dat_path).cube_matrix(first_nm=self.CONFIG_DATALOADER[DLK.FIRST_NM],
                                                                last_nm=self.CONFIG_DATALOADER[DLK.LAST_NM])

        return spectrum_data

    @staticmethod
    def mask_read(mask_path: str) -> np.ndarray:
        """ Load an Image

        Load an Image from a given path. There are only .PNG, .JPG, .JPEG and .JPE image format supported.
        If you load an JPG/JPEG/JPE image there will be an alpha channel added.

        :param mask_path: The path from the image to load.

        :return: Returns an 3D array with the RGBA color for every Pixel.

        :raise ValueError: For file not found or not supported image format.

        """
        from PIL import Image

        try:
            img = Image.open(mask_path)
        except Exception:
            raise FileNotFoundError("Mask file not found. Check your configurations!")

        if img.format not in ["PNG", "JPG", "JPEG", "JPE"]:
            raise ValueError("Mask format not supported! "
                             "Only '.png', '.jpeg', '.jpg' or '.jpe' are supported.")
        # add alpha channel
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            if img.format == "PNG":
                warnings.warn(".png without alpha channel. Alpha channel added.")
            else:
                warnings.warn("Better use '.png' format. Alpha channel added.")

        return np.reshape(img.getdata(), newshape=img.size[::-1] + (4,))

    def mk2_mask(self, mask_path: str, shape: tuple) -> np.ndarray:
        """ Loads Marker from .mk2 file and returns an annotation mask

        Read the .mk2 file and create a 3D array with the RGBA code for every classification.

        :param mask_path: File path for .mk2 file
        :param shape: Shape for the annotation mask

        :return: Returns a 3D array with the RGBA color for every pixel.
        """
        from data_utils.marker import MK2

        mk_loader = MK2(mask_path)
        names, leftx, topx, radiusx, indexx = mk_loader.load()

        class_mask = np.full(shape, -1)
        for idx in range(len(names)):
            if names[idx] == "":
                continue
            classification = -1
            for key, value in self.CONFIG_DATALOADER[DLK.TISSUE_LABELS].items():
                if names[idx].lower().replace(" ", "") == value.lower().replace(" ", ""):
                    classification = key
                    break

            radius = radiusx[idx]
            left = leftx[idx]
            top = topx[idx]

            x_ = np.arange(left - radius - 1, left + radius + 1, dtype=int)
            y_ = np.arange(top - radius - 1, top + radius + 1, dtype=int)
            # all pixel from circle
            x, y = np.where((x_[:, np.newaxis] - left) ** 2 + (y_ - top) ** 2 <= radius ** 2)
            for x_c, y_c in zip(x_[x], y_[y]):
                class_mask[y_c, x_c] = classification

        mask = np.zeros(shape + (4,))
        uni_classes = np.unique(class_mask)

        for uni_class in uni_classes:
            if uni_class != -1:
                color = self.CONFIG_DATALOADER[DLK.MASK_COLOR][uni_class][0].copy()
                if len(color) == 3:
                    color.append(255)
                else:
                    color[-1] = 255
                mask[class_mask == uni_class] = color
        # np.flipud is used because left and top start from left down to count
        return np.flipud(mask)
