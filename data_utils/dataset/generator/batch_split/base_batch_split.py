from typing import List, Dict, Tuple

import numpy as np
import os

from data_utils.data_storage import DataStorage
from configuration.parameter import (
    BATCH_FILE,
)


class BaseBatchSplit:
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

    def _split_and_save_batches(self, save_path: str, data, data_indexes: np.ndarray, rest) -> Dict[str, np.ndarray]:
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

            if arch[self.save_dict_names[0]].shape[0] < self.batch_size:
                return arch
            else:
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

    def _init_archives(self, batch_save_path: str, train_folder: str, valid_folder: str) -> Tuple[str, str]:
        train_dir = os.path.join(batch_save_path, train_folder)
        self.__init_archive(path=train_dir)
        valid_dir = os.path.join(batch_save_path, valid_folder)
        self.__init_archive(path=valid_dir)

        return train_dir, valid_dir

    def __init_archive(self, path: str):
        if not os.path.exists(path=path):
            os.mkdir(path)
        else:
            self.data_storage.delete_archive(delete_path=path)
            os.mkdir(path)

    @staticmethod
    def _check_dict_names(dict_names, data_):
        diff_dict_names = set(dict_names) - set(data_)
        diff_dataset = set(data_) - set(dict_names)

        if len(diff_dict_names):
            print(f'WARNING! dict_names {diff_dict_names} are not in dataset')
        if len(diff_dataset):
            print(f'WARNING! dict_names {diff_dataset} are not in dict_names of Preprocessor')

    @staticmethod
    def _check_name_in_data(indexes: np.ndarray, data_path: str, names: List[str]):
        if indexes.shape[0] == 0:
            print(f"WARING! No data found in {data_path} for the names: {','.join(n for n in names)}.")

    def _init_rest(self) -> Dict[str, np.ndarray]:
        rest_array = {}
        for name in self.dict_names:
            rest_array[name] = np.empty(0)
        return rest_array
