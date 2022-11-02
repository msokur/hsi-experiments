import config
from data_loader_base import DataLoader
from data_utils.hypercube_data import Cube_Read


class DataLoaderColon(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_extension(self):
        return config.FILE_EXTENSIONS['_dat']

    def get_labels(self):
        return super().get_labels()

    def get_name(self, path):
        return path.split(config.SYSTEM_PATHS_DELIMITER)[-1].split(".")[0].split('SpecCube')[0]

    def indexes_get_bool_from_mask(self, mask):
        healthy_indexes = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)  # blue
        ill_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)  # yellow
        not_certain_indexes = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)  # red

        return healthy_indexes, ill_indexes, not_certain_indexes

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        spectrum = DataLoaderColon.spectrum_read_from_dat(path)

        if mask_path is None:
            mask_path = path + '_Mask JW Kolo.png'
        mask = DataLoaderColon.mask_read(mask_path)

        return spectrum, mask

    def labeled_spectrum_get_from_dat(self, dat_path, mask_path=None):
        spectrum, mask = self.file_read_mask_and_spectrum(dat_path, mask_path=mask_path)
        healthy_indexes, ill_indexes, not_certain_indexes = self.indexes_get_bool_from_mask(mask)

        return spectrum[healthy_indexes], spectrum[ill_indexes], spectrum[not_certain_indexes]

    @staticmethod
    def spectrum_read_from_dat(dat_path):
        spectrum_data, _ = Cube_Read(dat_path,
                                     wavearea=config.WAVE_AREA,
                                     Firstnm=config.FIRST_NM,
                                     Lastnm=config.LAST_NM).cube_matrix()
        return spectrum_data

    @staticmethod
    def mask_read(mask_path):
        import cv2
        mask = cv2.imread(mask_path)[..., ::-1]  # [..., ::-1] - BGR to RGB
        return mask

    @staticmethod
    def labeled_spectrum_get_from_X_y(X, y):
        healthy_spectrum = X[y == 0]
        ill_spectrum = X[y == 1]
        not_certain_spectrum = X[y == 2]

        return healthy_spectrum, ill_spectrum, not_certain_spectrum
