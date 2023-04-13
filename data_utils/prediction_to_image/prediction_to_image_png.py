import numpy as np
import cv2

from data_utils.prediction_to_image.prediction_to_image_base import PredictionToImage_base


class PredictionToImage_png(PredictionToImage_base):
    def get_spectrum(self, path: str):
        pass

    def get_annotation_mask(self, path: str) -> np.ndarray:
        img = self.load_img(path=path)
        mask = self.whole_mask(img)

        return mask

    # TODO perhaps refactoring -> same code in data_utils.data_loaders.data_loaders_dyn.py
    @staticmethod
    def load_img(path: str) -> np.ndarray:
        """
                Loads an image and returns a numpy array with the RGBA Colorcode.

                :param path: The path from the image.

                :return: Returns an RGBA array from the Image.
                """
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # read Image with transparency
        # add alpha channel
        if mask.shape[-1] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

        # [..., -2::-1] - BGR to RGB, [..., -1:] - only transparency, '-1' - concatenate along last axis
        mask = np.r_["-1", mask[..., -2::-1], mask[..., -1:]]

        return mask

    def whole_mask(self, mask: np.array) -> np.ndarray:
        """
        Create a mask with all class labels. -1 is not used indexes.

        :param mask: Annotation mask with RGB or RGBA color code.

        :return: Returns an array with all class labels.

        Example
        -------
        >>> color = {0: [[255, 0, 0]], 1: [[0, 0, 255]]}
        >>> a = np.array(
        ...     [[[255, 0, 0], [255, 0, 0], [0, 255, 0]],
        ...      [[0, 255, 0], [0, 0, 255], [0, 255, 0]],
        ...      [[0, 0, 255], [0, 0, 255], [255, 0, 0]]])
        >>> b = self.whole_mask(a)
        array([ [0,     0,  -1],
        ...     [-1,    1,  -1],
        ...     [1,     1,   0]  ]

        """
        class_mask = np.full(mask.shape[0:2], -1)
        for key, value in self.load_conf["MASK_COLOR"].items():
            if key not in self.load_conf["LABELS_TO_TRAIN"]:
                continue

            for sub_value in value:
                if len(sub_value) == 3:
                    color_code = ((mask[..., 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                  (mask[:, :, 2] == sub_value[2]))
                elif len(sub_value) == 4:
                    color_code = ((mask[:, :, 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                  (mask[:, :, 2] == sub_value[2]) & (mask[:, :, 3] > sub_value[3]))
                else:
                    raise ValueError("Check your RGB/RGBA codes in configfile!")

                class_mask[color_code] = key

        return class_mask
