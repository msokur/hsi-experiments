from typing import List
import os

from data_utils.dataset import ChoiceNames
from data_utils.data_storage import DataStorage

from configuration.keys import (
    DataLoaderKeys as DLK,
    PreprocessorKeys as PPK,
    PathKeys as PK
)


class ExcludedPatients:
    def __init__(self, data_storage: DataStorage, config, log_dir: str, all_patients: List[str]):
        self.data_storage = data_storage
        self.config = config
        self.log_dir = log_dir
        self.all_patients = all_patients
        self.LEAVE_OUT_NAMES = None
        self.TRAIN_NAMES = None
        self.VALID_NAMES = None

    def set_names(self, leave_out_names: List[str], train_step_name: str):
        self.LEAVE_OUT_NAMES = leave_out_names
        choice_names = ChoiceNames(data_storage=self.data_storage, config_cv=self.config.CONFIG_CV,
                                   labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                   y_dict_name=self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1],
                                   log_dir=os.path.join(self.log_dir, train_step_name))

        self.VALID_NAMES = choice_names.get_valid_except_names(raw_path=self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
                                                               except_names=self.LEAVE_OUT_NAMES)

        self.TRAIN_NAMES = list(set(self.all_patients) - set(self.LEAVE_OUT_NAMES) - set(self.VALID_NAMES))

    def print_names(self):
        print(f"Leave out patient data: {', '.join(n for n in self.LEAVE_OUT_NAMES)}.\n")
        print(f"Train patient data: {', '.join(n for n in self.TRAIN_NAMES)}.\n")
        print(f"Valid patient data: {', '.join(n for n in self.VALID_NAMES)}.\n")
