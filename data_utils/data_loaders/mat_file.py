import os
import scipy.io as sio
import numpy as np
from glob import glob


class MatFile:
    def __init__(self, loader_conf: dict):
        self.loader = loader_conf

    def get_paths_and_splits(self, path):
        paths = glob(os.path.join(path, "*npz"))
        paths = self.sort(paths)

        if self.loader["CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"].dtype == int:
            splits = self.split_int(path_length=len(paths))
        else:
            splits = self.split_list(paths=paths)

        return paths, splits

    def split_int(self, path_length: int):
        cv_split = int(path_length / self.loader["CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"])
        splits = np.array_split(range(path_length), cv_split)

        return splits

    def split_list(self, paths):
        paths = np.array(paths)

        split_list = []
        for splits in self.loader["CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"]:
            split_part_list = []
            for split in splits:
                split_part_list.append(np.flatnonzero([True if split in path else False for path in paths])[0])
            split_part_list_np = np.array(split_part_list, dtype=np.uint8)
            split_list.append(split_part_list_np)

        split_list = np.array(split_list)

        return np.array_split(split_list, split_list.shape[0])

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for value in self.loader["LABELS"]:
            indexes.append((mask == value))

        return indexes

    def get_number(self, elem: str) -> str:
        return elem.split(self.loader["NUMBER_SPLIT"][0])[-1].split(".")[0].split(self.loader["NUMBER_SPLIT"][1])[0]

    def sort(self, paths):
        def take_only_number(elem):
            return int(self.get_number(elem=elem))

        paths = sorted(paths, key=take_only_number)

        return paths

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        data = sio.loadmat(path)
        spectrum, mask = data[self.loader["SPECTRUM"]], data[self.loader["MASK"]]

        return spectrum, mask
