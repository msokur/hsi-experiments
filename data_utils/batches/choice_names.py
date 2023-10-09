from typing import List, Union
from glob import glob

import numpy as np
import pickle
import os

from data_utils.data_archive import DataArchive
from configuration.keys import CrossValidationKeys as CVK
from configuration.parameter import (
    VALID_LOG,
)


class ChoiceNames:
    def __init__(self, data_archive: DataArchive, config_cv: dict, labels: Union[int, str], y_dict_name: str,
                 log_dir: str = None, set_seed=True):
        self.data_archive = data_archive
        self.CONFIG_CV = config_cv
        self.labels = labels
        self.y_dict_name = y_dict_name
        self.log_dir = log_dir
        if set_seed:
            np.random.seed(seed=1)

    @staticmethod
    def random_choice(paths, excepts, size=1) -> np.ndarray:
        return np.random.choice([r for r in paths if r not in excepts],
                                size=size,
                                replace=False)

    def class_choice(self, paths, paths_names, excepts, classes=None) -> np.ndarray:
        if classes is None:
            classes = np.array([])
        valid = self.random_choice(paths_names, excepts)

        path_idx = paths_names.index(valid[0])
        y = self.data_archive.get_data(data_path=paths[path_idx], data_name=self.y_dict_name)
        unique_classes = np.unique(y)
        con_classes = np.concatenate((classes, unique_classes))
        con_unique_classes = np.intersect1d(con_classes, self.labels)
        if len(con_unique_classes) >= len(self.labels):
            return valid
        elif len(con_unique_classes) - len(classes) >= 1:
            return np.concatenate((valid, self.class_choice(paths,
                                                            paths_names,
                                                            np.concatenate((excepts, valid)),
                                                            con_unique_classes)))
        else:
            return self.class_choice(paths,
                                     paths_names,
                                     np.concatenate((excepts, valid)),
                                     classes)

    def get_valid_except_names(self, raw_path: str, except_names: List[str]):
        if self.CONFIG_CV[CVK.CHOOSE_EXCLUDED_VALID] == "restore":
            print("Restore names of patients that will be used for validation dataset")
            restore_paths = glob(os.path.join(self.CONFIG_CV[CVK.RESTORE_VALID_PATH], "*", ""))
            restore_path = restore_paths[np.flatnonzero(
                np.core.defchararray.find(restore_paths, self.CONFIG_CV[CVK.RESTORE_VALID_SEQUENCE]) != -1)[0]]

            log_name = os.path.split(self.log_dir)[-1]
            log_index = log_name.split("_")[1]  # can be problems

            restore_log_paths = glob(os.path.join(restore_path, "*", ""))
            restore_log_path = restore_log_paths[
                np.flatnonzero(np.core.defchararray.find(restore_log_paths, "3d_" + str(log_index) + "_") != -1)[0]]

            valid_except_indexes = pickle.load(
                open(os.path.join(restore_log_path, VALID_LOG), "rb"))
            print(
                f"We restore {valid_except_indexes} from {restore_log_path} "
                f"with {self.CONFIG_CV[CVK.RESTORE_VALID_SEQUENCE]}")
            return valid_except_indexes

        raw_paths = self.data_archive.get_paths(archive_path=raw_path)
        raw_paths_names = [os.path.split(r)[-1].split(".")[0] for r in raw_paths]

        print('Getting new validation patients')
        if self.CONFIG_CV[CVK.CHOOSE_EXCLUDED_VALID] == "randomly":
            return self.random_choice(paths=raw_paths_names,
                                      excepts=except_names,
                                      size=self.CONFIG_CV[CVK.HOW_MANY_VALID_EXCLUDE])

        elif self.CONFIG_CV[CVK.CHOOSE_EXCLUDED_VALID] == "by_class":
            return self.class_choice(paths=raw_paths,
                                     paths_names=raw_paths_names,
                                     excepts=except_names)
