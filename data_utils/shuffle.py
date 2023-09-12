import os
import random
import pickle
from typing import List

import numpy as np

from glob import glob
from tqdm import tqdm

from configuration.copy_py_files import copy_files
from data_utils.data_archive.data_archive import DataArchive

from configuration.parameter import (
    SHUFFLE_GROUP_NAME, PILE_NAME
)


class Shuffle:
    def __init__(self, data_archive: DataArchive, raw_path: str, dict_names: list, piles_number: int, shuffle_saving_path: str, augmented=False,
                 files_to_copy: list = None):
        self.data_archive = data_archive
        self.raw_path = raw_path
        self.dict_names = dict_names
        self.piles_number = piles_number
        self.shuffle_saving_path = shuffle_saving_path
        self.augmented = augmented
        self.files_to_copy = files_to_copy

    def shuffle(self):
        print("--------Shuffling started--------")
        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(self.shuffle_saving_path, self.files_to_copy)

        piles = self.__create_piles()
        self.__split_into_piles(piles=piles)
        self.__shuffle_piles()
        print("--------Shuffling finished--------")

    # ------------------divide all samples into piles_number files------------------
    def __create_piles(self) -> List[list]:
        print("----Piles creating started----")

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

        for i, p in tqdm(enumerate(self.data_archive.get_paths(archive_path=self.raw_path))):
            # clear piles for new randon numbers
            for pn in range(self.piles_number):
                piles[pn] = []

            name = self.data_archive.get_name(path=p)
            _data = self.data_archive.get_datas(data_path=p)

            data = {n: a for n, a in _data.items()}
            X, y = data[self.dict_names[0]][...], data[self.dict_names[1]][...]

            if self.augmented:
                y = [[_y_] * X.shape[1] for _y_ in y]
                data[self.dict_names[0]] = np.concatenate(X, axis=0)
                data[self.dict_names[1]] = np.concatenate(y, axis=0)

            # fill random distribution to files
            for it in range(data[self.dict_names[0]].shape[0]):
                pile = random.randint(0, self.piles_number - 1)
                piles[pile].append(it)

            for i_pile, pile in enumerate(piles):
                _names = [name] * len(pile)
                _indexes = [i] * len(pile)

                values = {}
                for k in data.keys():
                    if k in self.dict_names:
                        values[k] = data[k][pile]

                values[self.dict_names[2]] = _names
                values[self.dict_names[3]] = _indexes

                pickle.dump(values, open(os.path.join(self.shuffle_saving_path, str(i_pile) + '.pile'), 'ab'))

        print("--Splitting into piles finished--")

    def __shuffle_piles(self):
        print("----Shuffling of piles started----")
        piles_paths = glob(os.path.join(self.shuffle_saving_path, f"*{PILE_NAME}"))
        print(len(piles_paths))

        for i, pp in tqdm(enumerate(piles_paths)):
            data = []
            with open(pp, "rb") as fr:
                try:
                    while True:
                        data.append(pickle.load(fr))
                except EOFError:
                    pass

            _data = {}
            for key in data[0].keys():
                _data[key] = [f[key] for f in data]
                _data[key] = np.concatenate(_data[key], axis=0)

            indexes = list(np.arange(_data[self.dict_names[0]].shape[0]))
            random.shuffle(indexes)

            os.remove(pp)
            self.data_archive.save_group(save_path=self.shuffle_saving_path, group_name=f"{SHUFFLE_GROUP_NAME}_{i}",
                                         datas={n: a[indexes] for n, a in _data.items()})

        print("----Shuffling of piles finished----")
