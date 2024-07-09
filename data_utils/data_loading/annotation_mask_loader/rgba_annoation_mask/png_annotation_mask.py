import warnings

import numpy as np

from .rgba_annotation_mask_interface import RGBAAnnotationMaskInterface


class PNGAnnotationMask(RGBAAnnotationMaskInterface):
    def __init__(self, config):
        super().__init__(config=config)

    def get_mask(self, mask_path: str, shape: tuple) -> np.ndarray:
        """ Load an Image

        Load an Image from a given path. There are only .PNG, .JPG, .JPEG and .JPE image format supported.
        If you load an JPG/JPEG/JPE image there will be an alpha channel added.

        :param mask_path: The path from the image to load.
        :param shape: The shape of the first two axis of the mask.

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

        mask = np.reshape(img.getdata(), newshape=img.size[::-1] + (4,))
        self._check_shapes(mask_path=mask_path,
                           mask_shape=mask.shape[:2],
                           expected_shape=shape)

        return mask

    @staticmethod
    def get_extension() -> str:
        return ".png"
