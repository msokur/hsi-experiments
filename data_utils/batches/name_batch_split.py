import abc
from typing import List, Dict
from tqdm import tqdm

import os
import numpy as np

from data_utils.data_archive import DataArchive
from configuration.parameter import (
    BATCH_FILE
)


class NameBatchSplit:
    def __init__(self, data_archive: DataArchive, batch_size: int, use_labels: List[int], dict_names: List[str],
                 with_sample_weights: bool):
        self.data_archive = data_archive
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.dict_names = dict_names
        self.X_dict_name = dict_names[0]
        self.label_dict_name = dict_names[1]
        self.p_dict_name = dict_names[2]
        self.weight_dict_name = dict_names[4]
        self.save_dict_names = [self.X_dict_name, self.label_dict_name]
        if with_sample_weights:
            self.save_dict_names.append(self.weight_dict_name)

    def split(self, data_paths: List[str], batch_save_path: str, except_names: List[str]) -> List[str]:
        print(f"--------Splitting {os.path.split(batch_save_path)[-1]} data into batches started--------")
        # ------------removing of previously generated archives (of the previous CV step) ----------------
        self.__init_archive__(path=batch_save_path)
        # ----- split datas into batches ----
        rest = self.__split_data_archive__(root_data_paths=data_paths,
                                           batch_save_path=batch_save_path,
                                           except_names=except_names)
        # ----- save rest of archive ----
        if len(rest[self.dict_names[0]]) >= self.batch_size:
            self.__split_and_save_batches__(save_path=batch_save_path, data={k: np.array(v) for k, v in rest.items()},
                                            data_indexes=np.full(shape=len(rest[self.dict_names[0]]), fill_value=True))

        print(f"--------Splitting {os.path.split(batch_save_path)[-1]} data into batches finished--------")
        return self.data_archive.get_paths(archive_path=batch_save_path)

    @abc.abstractmethod
    def __split_data_archive__(self, root_data_paths: List[str], batch_save_path: str,
                               except_names: List[str]) -> Dict[str, list]:
        rest = self.__init_rest()

        for p in tqdm(root_data_paths):
            # ------------ except_indexes filtering ---------------
            data_ = self.data_archive.get_datas(data_path=p)
            self.__check_dict_names(self.dict_names, data_)

            p_names = data_[self.p_dict_name][...]
            labels = data_[self.label_dict_name][...]

            # ------------ get only needed classes indexes --------------
            label_indexes = np.isin(labels, self.use_labels)

            # ------------ get data indexes --------------
            data_indexes = label_indexes & np.isin(p_names, except_names)
            self.__check_name_in_data(indexes=data_indexes, data_path=p, names=except_names)

            # ------------- split data ----------------
            rest_temp = self.__split_and_save_batches__(save_path=batch_save_path, data=data_,
                                                        data_indexes=data_indexes)

            # ---------------- save rest from archive ------------------
            for name in self.save_dict_names:
                rest[name] += list(rest_temp[name])

        return rest

    def __split_and_save_batches__(self, save_path: str, data, data_indexes: np.ndarray) -> Dict[str, np.ndarray]:
        # ---------------splitting into archives----------
        chunks = data_indexes.shape[0] // self.batch_size
        chunks_max = chunks * self.batch_size

        if chunks > 0:
            data_ = {k: np.array_split(data[k][...][data_indexes][:chunks_max], chunks) for k in self.save_dict_names}

            idx = len(self.data_archive.get_paths(archive_path=save_path))
            for row in range(chunks):
                arch = {}
                for i, n in enumerate(self.save_dict_names):
                    arch[n] = data_[n][row]

                self.data_archive.save_group(save_path=save_path, group_name=f"{BATCH_FILE}{idx}", datas=arch)
                idx += 1

        # ---------------saving of the non equal last part for the future partition---------
        rest = {k: data[k][...][data_indexes][chunks_max:] for k in self.save_dict_names}
        # ---------------saving of the non equal last part for the future partition---------
        return rest

    def __init_archive__(self, path: str):
        if not os.path.exists(path=path):
            os.mkdir(path)
        else:
            self.data_archive.delete_archive(delete_path=path)
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

    def __init_rest(self) -> Dict[str, list]:
        rest_array = {}
        for name in self.dict_names:
            rest_array[name] = []
        return rest_array
