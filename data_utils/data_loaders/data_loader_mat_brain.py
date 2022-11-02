import numpy as np
from glob import glob
import os
import scipy.io

import config
from data_loader_mat import DataLoaderMat


class DataLoaderMatBrain(DataLoaderMat):

    def get_number(self, elem):
        return elem.split("Op")[-1].split('.')[0].split('C')[0]

    def get_paths_and_splits(self, root_path=config.RAW_NPZ_PATH):
        paths = glob(os.path.join(root_path, '*.npz'))
        paths = self.sort(paths)
        paths = np.array(paths)

        p8c1 = np.flatnonzero([True if 'Op8C1' in path else False for path in paths])[0]
        p8c2 = np.flatnonzero([True if 'Op8C2' in path else False for path in paths])[0]
        p12c1 = np.flatnonzero([True if '12C1' in path else False for path in paths])[0]
        p12c2 = np.flatnonzero([True if '12C2' in path else False for path in paths])[0]
        p15c1 = np.flatnonzero([True if '15C1' in path else False for path in paths])[0]
        p20c1 = np.flatnonzero([True if '20C1' in path else False for path in paths])[0]

        splits = np.array(
            [np.array([p8c1, p8c2], dtype=np.uint8), np.array([p12c1, p12c2], dtype=np.uint8), [p15c1], [p20c1]])
        splits = np.array_split(splits, 4)

        return paths, splits

    def file_read_mask_and_spectrum(self, path):
        data = scipy.io.loadmat(path)
        spectrum, mask = data['preProcessedImage'], data['gtMap']

        return spectrum, mask

    def get_labels(self):
        return [0, 1, 2, 3]

    def indexes_get_bool_from_mask(self, mask):
        """> 0-- Unlabeled data
        > 1-- Normal (Healthy) tissue
        > 2-- Tumor tissue
        > 3-- Blood vessel
        > 4-- Background"""
        ill_indexes = (mask == 2)
        healthy_indexes = (mask == 1)
        vessel_indexes = (mask == 3)
        background_indexes = (mask == 4)

        return healthy_indexes, ill_indexes, vessel_indexes, background_indexes
