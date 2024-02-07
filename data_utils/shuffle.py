import abc
import os
import random
import pickle
from typing import List, Dict

import numpy as np

from glob import glob
from tqdm import tqdm
import warnings

from configuration.copy_py_files import copy_files
from data_utils.data_storage import DataStorage
from data_utils.dataset import save_tfr_file
from data_utils.dataset.meta_files import write_meta_info

from configuration.parameter import (
    SHUFFLE_GROUP_NAME, PILE_NAME, MAX_SIZE_PER_PILE
)


class Shuffle:
    def __init__(self, config, data_storage: DataStorage, raw_path: str, dict_names: list, dataset_typ: str,
                 set_seed: bool = True):
        self.data_storage = data_storage
        self.raw_path = raw_path
        self.config = config
        self.dict_names = dict_names
        self.piles_number = self.config.CONFIG_PREPROCESSOR["PILES_NUMBER"]
        self.shuffle_saving_path = self.config.CONFIG_PATHS["SHUFFLED_PATH"]
        self.dataset_typ = dataset_typ
        self.set_seed = set_seed

    def shuffle(self):
        print("--------Shuffling started--------")
        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(self.shuffle_saving_path,
                   self.config.CONFIG_PREPROCESSOR["FILES_TO_COPY"])

        if self.set_seed:
            random.seed(a=42)

        self.__check_piles_number()
        piles = self.__create_piles()
        self.__split_into_piles(piles=piles)
        self.__shuffle_piles()
        print("--------Shuffling finished--------")

    # ------------------divide all samples into piles_number files------------------

    def __check_piles_number(self):
        size = 0.0
        # get size from data archive
        for p in self.data_storage.get_paths(storage_path=self.raw_path):
            # get file size in bytes and convert to GB
            size += os.path.getsize(p) / (1024.0 ** 3)

        # add 2,5% more space for the patient index and name array
        size *= 1.025
        size_per_pile = size / self.piles_number

        # set new self.piles_number if the calculated pile size bigger den the maximum size
        if size_per_pile > MAX_SIZE_PER_PILE:
            # calculate the needed pile size
            new_piles_number = int(-(-size // MAX_SIZE_PER_PILE))
            warnings.warn(f"A maximum size of {MAX_SIZE_PER_PILE}GB per shuffle file is allowed, so the pile numer of "
                          f"{self.piles_number} is to small for a Dataset with the size {size}GB. Pile number is set to"
                          f" {new_piles_number}!")
            warnings.warn("If you need lager shuffle files change parameter 'MAX_SIZE_PER_PILE' in "
                          "configuration/parameter.")
            self.piles_number = new_piles_number

    def __create_piles(self) -> List[list]:
        print("----Piles creating started----")
        print(f"Pile number: {self.piles_number}")

        # remove previous piles if they exist
        piles_paths = glob(os.path.join(self.shuffle_saving_path, f"*{PILE_NAME}*"))
        for p in piles_paths:
            os.remove(p)

        # create clear piles
        piles = []
        for i in range(self.piles_number):
            piles.append([])
            open(os.path.join(self.shuffle_saving_path, f"{i}{PILE_NAME}"), "w").close()  # creating of an empty file

        print("----Piles creating finished----")
        return piles

    def __split_into_piles(self, piles: List[list]):
        print("--Splitting into piles started--")

        for i, p in tqdm(enumerate(self.data_storage.get_paths(storage_path=self.raw_path))):
            # clear piles for new randon numbers
            for pn in range(self.piles_number):
                piles[pn] = []

            name = self.data_storage.get_name(path=p)
            _data = self.data_storage.get_datas(data_path=p)

            data = {n: a for n, a in _data.items()}

            # fill random distribution to files
            for it in range(data[self.dict_names[0]].shape[0]):
                pile = random.randint(0, self.piles_number - 1)
                piles[pile].append(it)

            # get array with patient index and name
            for i_pile, pile in enumerate(piles):
                _names = [name] * len(pile)
                _indexes = [i] * len(pile)

                values = {}
                for k in data.keys():
                    if k in self.dict_names:
                        values[k] = data[k][...][pile]

                values[self.dict_names[2]] = _names
                values[self.dict_names[3]] = _indexes

                # save pile
                pickle.dump(values, open(os.path.join(self.shuffle_saving_path, str(i_pile) + PILE_NAME), 'ab'))

        print("--Splitting into piles finished--")
        del piles

    def __shuffle_piles(self):
        print("----Shuffling of piles started----")
        piles_paths = glob(os.path.join(self.shuffle_saving_path, f"*{PILE_NAME}"))

        # load all piles and there sub dictionary's
        for i, pp in tqdm(enumerate(piles_paths)):
            data = []
            with open(pp, "rb") as fr:
                try:
                    while True:
                        data.append(pickle.load(fr))
                except EOFError:
                    pass

            _data = {}
            # concatenate the subarray for every key from the datas in the pile
            for key in data[0].keys():
                _data[key] = [f[key] for f in data]
                _data[key] = np.concatenate(_data[key], axis=0)

            # shuffle the data in the pile
            indexes = list(np.arange(_data[self.dict_names[0]].shape[0]))
            random.shuffle(indexes)
            sh_data = {n: a[indexes] for n, a in _data.items()}

            # remove pile
            os.remove(pp)

            # save shuffled date and meta information
            sh_name = f"{SHUFFLE_GROUP_NAME}_{i}"
            write_meta_info(save_dir=self.shuffle_saving_path, file_name=sh_name,
                            labels=sh_data[self.dict_names[1]], names=sh_data[self.dict_names[2]],
                            names_idx=sh_data[self.dict_names[3]], X_shape=sh_data[self.dict_names[0]].shape,
                            typ=self.dataset_typ)

            self._save_file(file_name=sh_name, sh_data=sh_data)

        print("----Shuffling of piles finished----")

    @abc.abstractmethod
    def _save_file(self, file_name: str, sh_data: Dict[str, np.ndarray]):
        pass


class TFRShuffle(Shuffle):
    def _save_file(self, file_name: str, sh_data: Dict[str, np.ndarray]):
        save_tfr_file(save_path=self.shuffle_saving_path, file_name=file_name,
                      X=sh_data[self.dict_names[0]], y=sh_data[self.dict_names[1]],
                      pat_names=sh_data[self.dict_names[2]], pat_idx=sh_data[self.dict_names[3]],
                      idx_in_cube=sh_data[self.dict_names[4]], sw=sh_data[self.dict_names[5]])


class GeneratorShuffle(Shuffle):
    def _save_file(self, file_name: str, sh_data: Dict[str, np.ndarray]):
        self.data_storage.save_group(save_path=self.shuffle_saving_path, group_name=file_name, datas=sh_data)
