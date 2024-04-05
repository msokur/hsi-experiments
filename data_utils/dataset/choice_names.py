from typing import List, Union
from glob import glob

import numpy as np
import pickle
import os

from data_utils.data_storage import DataStorage
from configuration.keys import CrossValidationKeys as CVK
from configuration.parameter import (
    FILE_WITH_VALID_NAME,
)


class ChoiceNames:
    def __init__(self, data_storage: DataStorage, config_cv: dict, labels: List[Union[int, str]], y_dict_name: str,
                 log_dir: str = None, set_seed=True):
        """Class to choose file names.

        There are three options:

        - Restore -> restore names from a file in the given 'log_dir' with the sequent from the config
        - Random Choice -> take a random chose
        - Class Choice -> take a selection and looks if every label is in this selection

        :param data_storage: Object to load files
        :param config_cv: Configuration for typ of selection
        :param labels: The labels that should be in the selection (Class choice)
        :param y_dict_name: The array name in the file with the labels
        :param log_dir: Path to restore names (Restore)
        :param set_seed: If True it will set a seed for the randomness"""
        self.data_storage = data_storage
        self.CONFIG_CV = config_cv
        self.labels = labels
        self.y_dict_name = y_dict_name
        self.log_dir = log_dir
        if set_seed:
            np.random.seed(seed=1)

    @staticmethod
    def random_choice(paths, excepts, size=1) -> np.ndarray:
        """Choose a random value form an array.

        :param paths: Array with the values
        :param excepts: These values are not allowed to take
        :param size: How many values should be taken

        :return: A numpy array with the chosen values

        :raise ValueError: When there is no value in the array to take"""
        return np.random.choice([r for r in paths if r not in excepts],
                                size=size,
                                replace=False)

    def class_choice(self, paths, paths_names, excepts, classes=None) -> np.ndarray:
        """Take random file names and check if every label is in the whole selection.

        :param paths: The absolute file paths
        :param paths_names: File names in the same order from the paths
        :param excepts: File names that should not take
        :param classes: Don't care about these labels.

        :return: A numpy array with the chosen names

        :raises ValueError: When not abel to find all labels in the files"""
        if classes is None:
            classes = np.array([])
        # --- Check if there is any name left
        if (np.isin(paths_names, excepts)).all():
            raise ValueError("Check your data. Can't find enough files with all labels inside!")

        valid = self.random_choice(paths_names, excepts)

        path_idx = paths_names.index(valid[0])
        y = self.data_storage.get_data(data_path=paths[path_idx], data_name=self.y_dict_name)
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
            return self.restore_valid_names()

        raw_paths = self.data_storage.get_paths(storage_path=raw_path)
        raw_paths_names = [os.path.split(r)[-1].split(".")[0] for r in raw_paths]

        print('Generating random validation patients')
        if self.CONFIG_CV[CVK.CHOOSE_EXCLUDED_VALID] == "randomly":
            return self.random_choice(paths=raw_paths_names,
                                      excepts=except_names,
                                      size=self.CONFIG_CV[CVK.HOW_MANY_VALID_EXCLUDE])

        elif self.CONFIG_CV[CVK.CHOOSE_EXCLUDED_VALID] == "by_class":
            return self.class_choice(paths=raw_paths,
                                     paths_names=raw_paths_names,
                                     excepts=except_names)

    def restore_valid_names(self):
        print("Restoring of valid patients...")
        print('WARNING! Check if an order of the excepted test patients in the RESTORE_VALID_PATIENTS_FOLDER '
              'corresponds to the order of the excepted test patients in the current CV')

        log_name = os.path.split(self.log_dir)[-1]
        log_index = log_name.split("step_")[1]  # can be problems

        restore_log_paths = glob(os.path.join(self.CONFIG_CV[CVK.RESTORE_VALID_PATIENTS_FOLDER], "*", ""))
        restore_log_path = restore_log_paths[
            np.flatnonzero(np.core.defchararray.find(restore_log_paths, "step_" + str(log_index)) != -1)[0]]

        valid_except_indexes = pickle.load(open(os.path.join(restore_log_path, FILE_WITH_VALID_NAME), "rb"))
        print(f"We restore {valid_except_indexes} from {restore_log_path} ")
        return valid_except_indexes
