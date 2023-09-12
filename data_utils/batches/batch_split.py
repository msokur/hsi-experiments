import abc
from typing import List, Tuple
from tqdm import tqdm

import os
import numpy as np

from data_utils.data_archive.data_archive import DataArchive
from util.compare_distributions import DistributionsChecker
from configuration.parameter import (
    TRAIN, VALID, BATCH_FILE
)


class BatchSplit:
    def __init__(self, data_archive: DataArchive, batch_size: int, use_labels: List[int], dict_names: List[str],
                 config_distribution: dict):
        self.data_archive = data_archive
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.dict_names = dict_names
        self.spec_dict_name = dict_names[0]
        self.label_dict_name = dict_names[1]
        self.p_dict_name = dict_names[2]
        self.config_distribution = config_distribution

    def split(self, data_paths: List[str], batch_patch: str, except_cv_names: List[str], except_valid_names: List[str],
              tune=False) -> Tuple[str, str]:
        if tune:
            ds = DistributionsChecker(data_archive=self.data_archive, path=os.path.split(data_paths[0])[0],
                                      config_distribution=self.config_distribution, check_dict_name=self.spec_dict_name)
            tuning_index = ds.get_small_database_for_tuning()
            data_paths = [data_paths[tuning_index]]

        print('--------Splitting into npz of batch size started--------')
        # ------------removing of previously generated archives (of the previous CV step) ----------------
        self.__init_archive__(path=batch_patch)
        train_path, valid_path, except_train_names = self.__split_data_archive__(root_data_paths=data_paths,
                                                                                 root_batch_path=batch_patch,
                                                                                 except_cv_names=except_cv_names,
                                                                                 except_valid_names=except_valid_names)

        print(f"We except for patient out data: {','.join(n for n in except_cv_names)}.")
        print(f"We except for train data: {','.join(n for n in except_train_names)}.")
        print(f"We except for valid data: {','.join(n for n in except_valid_names)}.")
        print('--------Splitting into batches finished--------')
        return train_path, valid_path

    @abc.abstractmethod
    def __split_data_archive__(self, root_data_paths: List[str], root_batch_path: str, except_cv_names: List[str],
                               except_valid_names: List[str]) -> Tuple[str, str, List[str]]:
        arch_rest = self.__init_rest()
        valid_rest = self.__init_rest()
        except_names = except_cv_names + except_valid_names
        except_train_names = []
        train_path = os.path.join(root_batch_path, TRAIN)
        valid_path = os.path.join(root_batch_path, VALID)

        for p in tqdm(root_data_paths):
            # ------------ except_indexes filtering ---------------
            data_ = self.data_archive.get_datas(data_path=p)
            self.__check_dict_names(self.dict_names, data_)
            self.dict_names = list(set(self.dict_names).intersection(set(data_)))

            # data = {name: data_[name] for name in self.dict_names}
            p_names = data_[self.p_dict_name][...]
            labels = data_[self.label_dict_name][...]

            # ------------ get only needed classes indexes --------------
            label_indexes = np.isin(labels, self.use_labels)

            # ------------ get validation data indexes --------------
            valid_indexes = label_indexes & np.isin(p_names, except_valid_names)

            # ------------ get training data indexes --------------
            train_indexes = label_indexes & np.isin(p_names, except_names, invert=True)
            train_names = list(set(p_names[train_indexes]))
            except_train_names = list(set(except_train_names + train_names))
            if train_indexes.shape[0] == 0:
                print(f"WARING! No train data found in {p} for the names: {','.join(n for n in train_names)}.")

            # ------------- split train and valid data ----------------
            arch_rest_temp = self.data_archive.save_batch_arrays(save_path=train_path,
                                                                 data=data_,
                                                                 data_indexes=train_indexes,
                                                                 batch_file_name=BATCH_FILE,
                                                                 split_size=self.batch_size,
                                                                 save_dict_names=self.dict_names)
            valid_rest_temp = self.data_archive.save_batch_arrays(save_path=valid_path,
                                                                  data=data_,
                                                                  data_indexes=valid_indexes,
                                                                  batch_file_name=BATCH_FILE,
                                                                  split_size=self.batch_size,
                                                                  save_dict_names=self.dict_names)

            # ---------------- save rest from archive an valid data ------------------
            for name in self.dict_names:
                arch_rest[name] += list(arch_rest_temp[name])
                valid_rest[name] += list(valid_rest_temp[name])

        if len(arch_rest[self.dict_names[0]]) >= self.batch_size:
            self.data_archive.save_batch_arrays(save_path=train_path,
                                                data=arch_rest,
                                                data_indexes=np.full(shape=len(arch_rest[self.dict_names[0]]),
                                                                     fill_value=True),
                                                batch_file_name=BATCH_FILE,
                                                split_size=self.batch_size,
                                                save_dict_names=self.dict_names)

        if len(valid_rest[self.dict_names[0]]) >= self.batch_size:
            self.data_archive.save_batch_arrays(save_path=valid_path,
                                                data=valid_rest,
                                                data_indexes=np.full(shape=len(valid_rest_rest[self.dict_names[0]]),
                                                                     fill_value=True),
                                                batch_file_name=BATCH_FILE,
                                                split_size=self.batch_size,
                                                save_dict_names=self.dict_names)

        return train_path, valid_path, except_train_names

    def __init_archive__(self, path: str):
        self.__check_archive__(path=path)
        self.__check_archive__(path=os.path.join(path, TRAIN), delete_if_exist=True)
        self.__check_archive__(path=os.path.join(path, VALID), delete_if_exist=True)

    def __check_archive__(self, path: str, delete_if_exist=False):
        if not os.path.exists(path=path):
            os.mkdir(path)
        elif delete_if_exist:
            self.data_archive.delete_archive(delete_path=path)

    @staticmethod
    def __check_dict_names(dict_names, data_):
        diff_dict_names = set(dict_names) - set(data_)
        diff_dataset = set(data_) - set(dict_names)

        if len(diff_dict_names):
            print(f'WARNING! dict_names {diff_dict_names} are not in dataset')
        if len(diff_dataset):
            print(f'WARNING! dict_names {diff_dataset} are not in dict_names of Preprocessor')

    def __init_rest(self):
        rest_array = {}
        for name in self.dict_names:
            rest_array[name] = []
        return rest_array
