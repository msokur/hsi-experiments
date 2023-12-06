import os
import random
import pickle
from typing import List

import numpy as np

from glob import glob
from tqdm import tqdm
import warnings

from configuration.copy_py_files import copy_files
from data_utils.data_archive import DataArchive
from data_utils.tfrecord import write_meta_info, save_tfr_file

from configuration.parameter import (
    SHUFFLE_GROUP_NAME, PILE_NAME, MAX_SIZE_PER_PILE
)


class Shuffle:
    def __init__(self, data_archive: DataArchive, raw_path: str, dict_names: list, piles_number: int,
                 shuffle_saving_path: str, augmented=False, files_to_copy: list = None):
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

        if self.files_to_copy is not None:
            copy_files(self.shuffle_saving_path, self.files_to_copy)

        self.__check_piles_number()
        piles = self.__create_piles()
        self.__split_into_piles(piles=piles)
        self.__shuffle_piles()
        print("--------Shuffling finished--------")

    # ------------------divide all samples into piles_number files------------------

    def __check_piles_number(self):
        size = 0.0
        # get size from data archive
        for p in self.data_archive.get_paths(archive_path=self.raw_path):
            # get file size in bytes and convert to GB
            size += os.path.getsize(p) / (1024.0 ** 3)

        # add 1% more space for the patient index and name array
        size *= 1.01
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

            if self.augmented:
                X, y = data[self.dict_names[0]][...], data[self.dict_names[1]][...]
                y = [[_y_] * X.shape[1] for _y_ in y]
                data[self.dict_names[0]] = np.concatenate(X, axis=0)
                data[self.dict_names[1]] = np.concatenate(y, axis=0)

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

    def __shuffle_piles(self):
        print("----Shuffling of piles started----")
        piles_paths = glob(os.path.join(self.shuffle_saving_path, f"*{PILE_NAME}"))
        print(len(piles_paths))

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
            write_meta_info(save_dir=self.shuffle_saving_path, file_name=f"{SHUFFLE_GROUP_NAME}_{i}",
                            labels=sh_data[self.dict_names[1]], names=sh_data[self.dict_names[2]],
                            names_idx=sh_data[self.dict_names[3]], X_shape=sh_data[self.dict_names[0]].shape)
            save_tfr_file(save_path=self.shuffle_saving_path, file_name=f"{SHUFFLE_GROUP_NAME}_{i}",
                          X=sh_data[self.dict_names[0]], y=sh_data[self.dict_names[1]],
                          pat_names=sh_data[self.dict_names[2]], pat_idx=sh_data[self.dict_names[3]],
                          idx_in_cube=sh_data[self.dict_names[4]], sw=sh_data[self.dict_names[5]])

        print("----Shuffling of piles finished----")
