import os
import random

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

import zarr

from configuration.copy_py_files import copy_files

from configuration.parameter import (
    SHUFFLE_ARCHIVE
)


class ShuffleZarr:
    def __init__(self, raw_paths: List[str], dict_names: list, shuffle_saving_path: str, piles_number: int,
                 chunks: Tuple[int], files_to_copy: list = None):
        self.raw_paths = raw_paths
        self.raw_root_path = os.path.split(self.raw_paths[0])[0]
        self.dict_names = dict_names
        self.shuffle_saving_path = shuffle_saving_path
        self.piles_number = piles_number
        self.chunks = chunks
        self.files_to_copy = files_to_copy

    def shuffle(self):
        print("--------Shuffling started--------")
        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(self.shuffle_saving_path, self.files_to_copy)

        self.__shuffle_pat__()
        self.__create_shuffled_data__()

        print("--------Shuffling finished--------")

    def __shuffle_pat__(self):
        raw_archive = zarr.open_group(store=self.raw_paths[0], mode="r")
        shuffle_root = zarr.open_group(store=os.path.join(self.shuffle_saving_path, SHUFFLE_ARCHIVE), mode="a")
        # self.create_empty_shuffled_archive(shuffle_root=shuffle_root, raw_archive=raw_archive)

        print("--------Shuffling patient data started--------")
        for pat_idx, path in tqdm(enumerate(self.raw_paths)):
            name = os.path.split(path)[-1]
            data_group = zarr.open_group(store=path, mode="r")
            arr_names = [n for n in data_group.array_keys()]
            size = data_group[arr_names[0]].shape[0]

            piles = []
            for i in range(self.piles_number):
                piles.append([])

            for idx in range(size):
                pile = random.randint(0, self.piles_number - 1)
                piles[pile].append(idx)

            for idx, pile in enumerate(piles):
                shuffle_group = shuffle_root.get(f"{ZARR_SHUFFLE_GROUP}_{idx}")
                for arr_name in arr_names:
                    shuffle_group.get(arr_name).append(data_group.get(arr_name)[...][pile], axis=0)
                shuffle_group.get(self.dict_names[2]).append([name] * len(pile), axis=0)
                shuffle_group.get(self.dict_names[3]).append([pat_idx] * len(pile))

        print("--------Shuffling patient data finished--------")

    def __create_shuffled_data__(self):
        print("--------Create shuffling data started--------")

        shuffle_root = zarr.open_group(store=os.path.join(self.shuffle_saving_path, SHUFFLE_ARCHIVE), mode="r+")

        for index in tqdm(range(self.piles_number)):
            shuffle_group = shuffle_root.get(f"{ZARR_SHUFFLE_GROUP}_{index}")
            data_indexes = list(range(shuffle_group.get(self.dict_names[0]).shape[0]))
            random.shuffle(data_indexes)
            for key in shuffle_group.array_keys():
                data = shuffle_group[key]
                data[...] = data[...][data_indexes]

        print("--------Create shuffling data finished--------")

    def create_empty_shuffled_archive(self, shuffle_root: zarr.Group, raw_archive: zarr.Group):
        print("--------Create empty archives started--------")
        for idx in range(self.piles_number):
            shuffle_group = shuffle_root.create_group(name=f"{ZARR_SHUFFLE_GROUP}_{idx}", overwrite=True)
            for key in raw_archive.array_keys():
                shape = list(raw_archive[key].shape)
                shape[0] = 0
                shuffle_group.empty(name=key, shape=shape, chunks=self.chunks, dtype=raw_archive[key].dtype)

            # add empty array for Patient name
            shuffle_group.empty(name=self.dict_names[2], shape=0, chunks=self.chunks, dtype=str)
            # add empty array for Patient index
            shuffle_group.empty(name=self.dict_names[3], shape=0, chunks=self.chunks, dtype=int)
        print("--------Create empty archives finished--------")

    def temp_values(self, keys: List[str]) -> List[Dict[str, list]]:
        piles = []
        for i in range(self.piles_number):
            pile_dict = {}
            for key in keys:
                pile_dict[key] = []
            piles.append(pile_dict)

        return piles
