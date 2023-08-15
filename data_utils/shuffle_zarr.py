import os
import random
from tqdm import tqdm
from typing import List
import math

import zarr

from configuration.copy_py_files import copy_files

from configuration.parameter import (
    ZARR_SHUFFLE_GROUP, ZARR_SHUFFLE_NAME_ARRAY, ZARR_SHUFFLE_IDX_ARRAY
)


class ShuffleZarr:
    def __init__(self, raw_paths: List[str], shuffle_saving_path: str, piles_number: int, files_to_copy: list = None):
        self.raw_paths = raw_paths
        self.shuffle_saving_path = shuffle_saving_path
        self.piles_number = piles_number
        self.files_to_copy = files_to_copy

    def shuffle(self):
        print("--------Shuffling started--------")
        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(self.shuffle_saving_path, self.files_to_copy)

        piles = self.__shuffle_pat__()
        self.__create_shuffled_data__(piles=piles)

        print("--------Shuffling finished--------")

    def __shuffle_pat__(self) -> List[list]:
        print("--------Shuffling patient data started--------")
        piles = []
        for p in range(self.piles_number):
            piles.append([])

        for data in tqdm(self.raw_paths):
            name = os.path.split(data)[-1]
            data_group = zarr.open_group(store=data, mode="r")
            arr_names = [n for n in data_group.array_keys()]
            size = data_group[arr_names[0]].shape[0]

            for idx in range(size):
                pile = random.randint(0, self.piles_number - 1)
                piles[pile].append([name, idx])

        print("--------Shuffling patient data finished--------")
        return piles

    def __create_shuffled_data__(self, piles: List[list]):
        print(piles)
        print("--------Create shuffling data started--------")
        names = []
        idx = []

        for pile in tqdm(piles):
            random.shuffle(pile)
            for p in pile:
                names.append(p[0])
                idx.append(p[1])

        chunks = math.floor(len(names) / self.piles_number)

        shuffle_root = zarr.open_group(store=os.path.join(self.shuffle_saving_path, ZARR_SHUFFLE_GROUP), mode="w")
        shuffle_root.array(name=ZARR_SHUFFLE_NAME_ARRAY, data=names, chunks=chunks, dtype=str)
        shuffle_root.array(name=ZARR_SHUFFLE_IDX_ARRAY, data=idx, chunks=chunks)
        print("--------Create shuffling data finished--------")
