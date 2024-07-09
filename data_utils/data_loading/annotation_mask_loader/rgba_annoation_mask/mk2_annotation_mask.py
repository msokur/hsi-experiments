import numpy as np

from .rgba_annotation_mask_interface import RGBAAnnotationMaskInterface

from configuration.keys import (
    DataLoaderKeys as DLK
)


class Mk2AnnotationMask(RGBAAnnotationMaskInterface):
    def __init__(self, config):
        super().__init__(config=config)

    def get_mask(self, mask_path: str, shape: tuple) -> np.ndarray:
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
            for key, value in self.config.CONFIG_DATALOADER[DLK.TISSUE_LABELS].items():
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
                color = self.config.CONFIG_DATALOADER[DLK.MASK_COLOR][uni_class][0].copy()
                if len(color) == 3:
                    color.append(255)
                else:
                    color[-1] = 255
                mask[class_mask == uni_class] = color
        # np.flipud is used because left and top start from left down to count
        return np.flipud(mask)

    @staticmethod
    def get_extension() -> str:
        return ".mk2"
