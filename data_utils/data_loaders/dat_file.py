import os
import numpy as np
import cv2
from data_utils.hypercube_data import Cube_Read
from data_utils.marker import MK2


class DatFile:
    def __init__(self, loader_conf: dict):
        self.loader = loader_conf

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for key, value in self.loader["MASK_COLOR"].items():
            sub_mask = np.zeros(mask.shape[:2]).astype(dtype=bool)
            for sub_value in value:
                if len(sub_value) < 4:
                    sub_mask |= ((mask[..., 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                 (mask[:, :, 2] == sub_value[2]))
                else:
                    sub_mask |= ((mask[:, :, 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                 (mask[:, :, 2] == sub_value[2]) & (mask[:, :, 3] > sub_value[3]))
            indexes.append(sub_mask)

        return indexes

    def set_mask_with_label(self, mask):
        result_mask = np.zeros(mask.shape[:2]) - 1
        indexes = self.indexes_get_bool_from_mask(mask)
        for sub_mask, key in zip(indexes, self.loader["MASK_COLOR"].keys()):
            result_mask[sub_mask] = key

        return result_mask

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        spectrum = self.spectrum_read_from_dat(path)

        path_parts = os.path.split(path)
        name = path_parts[1].split(self.loader["MASK_DIFF"][0])[0] + self.loader["MASK_DIFF"][1]

        if mask_path is None:
            mask_path = os.path.join(path_parts[0], name)
        else:
            mask_path = os.path.join(mask_path, name)

        if mask_path.endswith(".mk2"):
            mask = self.mk2_mask(mask_path, spectrum.shape[0:2])
        else:
            mask = DatFile.mask_read(mask_path)

        return spectrum, mask

    @staticmethod
    def sort(paths):
        return sorted(paths)

    def spectrum_read_from_dat(self, dat_path):
        spectrum_data, _ = Cube_Read(dat_path,
                                     wavearea=self.loader["WAVE_AREA"],
                                     Firstnm=self.loader["FIRST_NM"],
                                     Lastnm=self.loader["LAST_NM"]).cube_matrix()

        return spectrum_data

    @staticmethod
    def mask_read(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # read Image with transparency
        # add alpha channel
        if mask.shape[-1] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

        # [..., -2::-1] - BGR to RGB, [..., -1:] - only transparency, '-1' - concatenate along last axis
        mask = np.r_["-1", mask[..., -2::-1], mask[..., -1:]]

        return mask

    def mk2_mask(self, mask_path, shape):
        mk_loader = MK2(mask_path)
        names, leftx, topx, radiusx, indexx = mk_loader.load()

        class_mask = np.full(shape, -1)
        for idx in range(len(names) - 1):
            classification = -1
            for key, value in self.loader["TISSUE_LABELS"].items():
                if names[idx].lower() == value.lower():
                    classification = key
                    break

            radius = radiusx[idx]
            left = leftx[idx]
            top = topx[idx]

            x_ = np.arange(left - radius - 1, left + radius + 1, dtype=int)
            y_ = np.arange(top - radius -1, top + radius + 1, dtype=int)
            # alle Pixel aus dem Kreis
            x, y = np.where((x_[:, np.newaxis] - left) ** 2 + (y_ - top) ** 2 <= radius ** 2)

            for x_c, y_c in zip(x_[x], y_[y]):
                class_mask[y_c, x_c] = classification

        mask = np.zeros(shape + (4,))
        uni_classes = np.unique(class_mask)

        for uni_class in uni_classes:
            if uni_class != -1:
                color = self.loader["MASK_COLOR"][uni_class][0].copy()
                color.append(255)
                mask[class_mask == uni_class] = color

        return np.flipud(mask)


if __name__ == "__main__":
    from configuration.get_config import DATALOADER
    dat_path_ = r"E:\ICCAS\Gastric\General\Laura_Daten0.dat"
    mk2_path = r"E:\ICCAS\Gastric\General\Registrations\mk_files\Laura_Daten0.mk2"

    dat_loader = DatFile(DATALOADER)

    spec, mask_ = dat_loader.file_read_mask_and_spectrum(dat_path_, mk2_path)
    bool_mask = dat_loader.indexes_get_bool_from_mask(mask_)
    x__ = 1
