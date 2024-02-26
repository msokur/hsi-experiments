from typing import List, Dict, Tuple
from tqdm import tqdm

import os
import numpy as np

from data_utils.data_storage import DataStorage
from ..utils import parse_names_to_int
from configuration.parameter import (
    BATCH_FILE
)


class NameBatchSplit:
    def __init__(self, data_storage: DataStorage, batch_size: int, use_labels: List[int], dict_names: List[str],
                 with_sample_weights: bool):
        self.data_storage = data_storage
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.dict_names = dict_names
        self.X_dict_name = dict_names[0]
        self.label_dict_name = dict_names[1]
        self.p_dict_name = dict_names[2]
        self.p_dict_name_idx = dict_names[3]
        self.weight_dict_name = dict_names[5]
        self.save_dict_names = [self.X_dict_name, self.label_dict_name]
        if with_sample_weights:
            self.save_dict_names.append(self.weight_dict_name)

    def split(self, data_paths: List[str], batch_save_path: str, train_names: List[str],
              valid_names: List[str], train_folder: str, valid_folder: str) -> Tuple[List[str], List[str]]:
        print(f"--------Splitting data into train and valid batches started--------")
        # ------------removing of previously generated archives (of the previous CV step) ----------------
        train_dir = os.path.join(batch_save_path, train_folder)
        self.__init_archive__(path=train_dir)
        valid_dir = os.path.join(batch_save_path, valid_folder)
        self.__init_archive__(path=valid_dir)
        # ----- split datas into batches ----
        self.__split_data_archive__(root_data_paths=data_paths,
                                    batch_train_save_path=train_dir,
                                    batch_valid_save_path=valid_dir,
                                    train_names=train_names,
                                    valid_names=valid_names)

        print(f"--------Splitting data into train and valid batches finished--------")
        train_paths = self.data_storage.get_paths(storage_path=train_dir)
        valid_paths = self.data_storage.get_paths(storage_path=valid_dir)
        return train_paths, valid_paths

    def __split_data_archive__(self, root_data_paths: List[str], batch_train_save_path: str, batch_valid_save_path: str,
                               train_names: List[str], valid_names: List[str]):
        train_rest = self.__init_rest()
        valid_rest = self.__init_rest()
        train_names_idx = self.__get_names_int(paths=root_data_paths, names=train_names)
        valid_names_idx = self.__get_names_int(paths=root_data_paths, names=valid_names)

        for p in tqdm(root_data_paths):
            # ------------ except_indexes filtering ---------------
            data_ = self.data_storage.get_datas(data_path=p)
            self.__check_dict_names(self.dict_names, data_)

            p_names_idx = data_[self.p_dict_name_idx][...]
            labels = data_[self.label_dict_name][...]

            # ------------ get only needed classes indexes --------------
            label_indexes = np.isin(labels, self.use_labels)

            # ------------ get data indexes --------------
            train_data_indexes = label_indexes & np.isin(p_names_idx, train_names_idx)
            self.__check_name_in_data(indexes=train_data_indexes, data_path=p, names=train_names)
            valid_data_indexes = label_indexes & np.isin(p_names_idx, valid_names_idx)
            self.__check_name_in_data(indexes=valid_data_indexes, data_path=p, names=train_names)

            # ------------- split data ----------------
            train_rest = self.__split_and_save_batches__(save_path=batch_train_save_path, data=data_,
                                                         data_indexes=train_data_indexes, rest=train_rest)
            valid_rest = self.__split_and_save_batches__(save_path=batch_valid_save_path, data=data_,
                                                         data_indexes=valid_data_indexes, rest=valid_rest)

    def __split_and_save_batches__(self, save_path: str, data, data_indexes: np.ndarray, rest) -> Dict[str, np.ndarray]:
        mask_data = {k: data[k][...][data_indexes] for k in self.save_dict_names}
        idx = len(self.data_storage.get_paths(storage_path=save_path))
        arch = {}

        rest_shape = rest[self.dict_names[1]].shape[0]
        if rest_shape == 0:
            batch_diff = 0
        else:
            batch_diff = self.batch_size - rest_shape

            # Save rest from the data before
            for n in self.save_dict_names:
                arch[n] = np.concatenate((rest[n], mask_data[n][:batch_diff]))

            self.data_storage.save_group(save_path=save_path, group_name=f"{BATCH_FILE}{idx}", datas=arch)
            idx += 1

        # ---------------splitting into archives----------
        chunks = (data_indexes.sum() - batch_diff) // self.batch_size
        chunks_max = chunks * self.batch_size + batch_diff

        if chunks > 0:
            data_ = {k: np.array_split(mask_data[k][batch_diff:chunks_max], chunks) for k in self.save_dict_names}

            for row in range(chunks):
                for n in self.save_dict_names:
                    arch[n] = data_[n][row]

                self.data_storage.save_group(save_path=save_path, group_name=f"{BATCH_FILE}{idx}", datas=arch)
                idx += 1

        # ---------------saving of the non-equal last part for the future partition---------
        rest = {k: mask_data[k][chunks_max:] for k in self.save_dict_names}
        # ---------------saving of the non-equal last part for the future partition---------
        return rest

    def __init_archive__(self, path: str):
        if not os.path.exists(path=path):
            os.mkdir(path)
        else:
            self.data_storage.delete_archive(delete_path=path)
            os.mkdir(path)

    @staticmethod
    def __check_dict_names(dict_names, data_):
        diff_dict_names = set(dict_names) - set(data_)
        diff_dataset = set(data_) - set(dict_names)

        if len(diff_dict_names):
            print(f'WARNING! dict_names {diff_dict_names} are not in dataset')
        if len(diff_dataset):
            print(f'WARNING! dict_names {diff_dataset} are not in dict_names of Preprocessor')

    @staticmethod
    def __check_name_in_data(indexes: np.ndarray, data_path: str, names: List[str]):
        if indexes.shape[0] == 0:
            print(f"WARING! No data found in {data_path} for the names: {','.join(n for n in names)}.")

    def __init_rest(self) -> Dict[str, np.ndarray]:
        rest_array = {}
        for name in self.dict_names:
            rest_array[name] = np.empty(0)
        return rest_array

    @staticmethod
    def __get_names_int(paths: list, names: list):
        names_int_dict = parse_names_to_int(files=paths, meta_type="generator")
        names_int = []
        for name in names:
            if name in names_int_dict:
                names_int += [names_int_dict[name]]

        return names_int
