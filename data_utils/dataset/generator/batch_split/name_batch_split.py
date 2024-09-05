from typing import List, Tuple
from tqdm import tqdm

import os
import numpy as np

from data_utils.data_storage import DataStorage
from data_utils.dataset.utils import parse_names_to_int
from .base_batch_split import BaseBatchSplit


class NameBatchSplit(BaseBatchSplit):
    def __init__(self, data_storage: DataStorage, batch_size: int, use_labels: List[int], dict_names: List[str],
                 with_sample_weights: bool):
        super().__init__(data_storage, batch_size, use_labels, dict_names, with_sample_weights)

    def split(self, data_paths: List[str], batch_save_path: str, train_names: List[str],
              valid_names: List[str], train_folder: str, valid_folder: str) -> Tuple[List[str], List[str]]:
        print(f"--------Splitting data into train and valid batches started--------")
        # ------------removing of previously generated archives (of the previous CV step) ----------------
        train_dir, valid_dir = self._init_archives(batch_save_path=batch_save_path,
                                                   train_folder=train_folder,
                                                   valid_folder=valid_folder)

        # ----- split datas into batches ----
        self.__split_data_archive(root_data_paths=data_paths,
                                  batch_train_save_path=train_dir,
                                  batch_valid_save_path=valid_dir,
                                  train_names=train_names,
                                  valid_names=valid_names)

        print(f"--------Splitting data into train and valid batches finished--------")
        train_paths = self.data_storage.get_paths(storage_path=train_dir)
        valid_paths = self.data_storage.get_paths(storage_path=valid_dir)
        return train_paths, valid_paths

    def __split_data_archive(self, root_data_paths: List[str], batch_train_save_path: str, batch_valid_save_path: str,
                             train_names: List[str], valid_names: List[str]):
        train_rest = self._init_rest()
        valid_rest = self._init_rest()
        train_names_idx = self.__get_names_int(paths=root_data_paths, names=train_names)
        valid_names_idx = self.__get_names_int(paths=root_data_paths, names=valid_names)

        for p in tqdm(root_data_paths):
            # ------------ except_indexes filtering ---------------
            data_ = self.data_storage.get_datas(data_path=p)
            self._check_dict_names(self.dict_names, data_)

            p_names_idx = data_[self.p_dict_name_idx][...]
            labels = data_[self.label_dict_name][...]

            # ------------ get only needed classes indexes --------------
            label_indexes = np.isin(labels, self.use_labels)

            # ------------ get data indexes --------------
            train_data_indexes = label_indexes & np.isin(p_names_idx, train_names_idx)
            self._check_name_in_data(indexes=train_data_indexes, data_path=p, names=train_names)
            valid_data_indexes = label_indexes & np.isin(p_names_idx, valid_names_idx)
            self._check_name_in_data(indexes=valid_data_indexes, data_path=p, names=train_names)

            # ------------- split data ----------------
            train_rest = self._split_and_save_batches(save_path=batch_train_save_path, data=data_,
                                                      data_indexes=train_data_indexes, rest=train_rest)
            valid_rest = self._split_and_save_batches(save_path=batch_valid_save_path, data=data_,
                                                      data_indexes=valid_data_indexes, rest=valid_rest)

    @staticmethod
    def __get_names_int(paths: list, names: list):
        names_int_dict = parse_names_to_int(files=paths, meta_type="generator")
        names_int = []
        for name in names:
            if name in names_int_dict:
                names_int += [names_int_dict[name]]

        return names_int
