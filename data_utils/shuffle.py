import abc
import os
import random
import pickle
from typing import List, Dict
import json

import numpy as np

from glob import glob
from tqdm import tqdm
import warnings

from configuration.copy_py_files import copy_files
from data_utils.data_storage import DataStorage
from data_utils.dataset import save_tfr_file
from data_utils.dataset.meta_files import write_meta_info

from configuration.keys import PreprocessorKeys as PPK, PathKeys as PK, TrainerKeys as TK, DataLoaderKeys as DK 
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
        self.piles_number = self.config.CONFIG_PREPROCESSOR[PPK.PILES_NUMBER]
        self.shuffle_saving_path = self.config.CONFIG_PATHS[PK.SHUFFLED_PATH]
        self.dataset_typ = dataset_typ
        self.set_seed = set_seed

    def shuffle(self, use_piles: List[int] = None):
        if use_piles is not None:
            use_piles = [pile_idx for pile_idx in use_piles if pile_idx < self.piles_number - 1]
        else:
            use_piles = self.__all_pile_idx()

        print("--------Shuffling started--------")
        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(self.shuffle_saving_path,
                   self.config.CONFIG_PREPROCESSOR["FILES_TO_COPY"])

        if self.set_seed:
            random.seed(a=42)

        if self.config.CONFIG_TRAINER[TK.USE_SMALLER_DATASET] and \
                PPK.SMALL_REPRESENTATIVE_DATASET in self.config.CONFIG_PREPROCESSOR and \
                self.config.CONFIG_PREPROCESSOR[PPK.SMALL_REPRESENTATIVE_DATASET]:
            self.__create_one_shuffled_archive_like_example()
        else:
            self.check_piles_number()
            self.__create_piles(use_piles=use_piles)
            self.__split_into_piles(use_piles=use_piles)
            self.__shuffle_piles()
        print("--------Shuffling finished--------")

    # ------------------divide all samples into piles_number files------------------

    def __create_one_shuffled_archive_like_example(self):
        import pandas as pd

        def get_name(string):
            return string.split(self.config.CONFIG_PATHS[PK.SYS_DELIMITER])[-1].split('.')[0]

        def adjust_meta_file():
            meta_name = example_name + '.meta'
            meta_data = json.load(open(file=os.path.join(example_root, meta_name), mode="r"))
            meta_data['X_shape'] = [*self.config.CONFIG_DATALOADER[DK.D3_SIZE],
                                    example['X'].shape[-1]]

            with open(os.path.join(self.shuffle_saving_path, meta_name), "w") as file:
                json.dump(meta_data, file)

        def print_number_of_zeros(text):
            shape = example['X'].shape
            reshaped_for_counting = np.reshape(example['X'], [shape[0], np.prod(shape[1:])])
            sums = np.sum(reshaped_for_counting, axis=-1)
            print(f"Number of unfilled samples ({text})", len(sums[sums == 0]))

            return sums != 0

        self.__remove_files(PILE_NAME)
        self.__remove_files("shuffled")

        example = dict(np.load(self.config.CONFIG_PREPROCESSOR[PPK.SMALL_REPRESENTATIVE_DATASET]))
        example_name = get_name(self.config.CONFIG_PREPROCESSOR[PPK.SMALL_REPRESENTATIVE_DATASET])
        example_root = os.path.dirname(self.config.CONFIG_PREPROCESSOR[PPK.SMALL_REPRESENTATIVE_DATASET])
        example['X'] = np.zeros([example['X'].shape[0],
                                 *self.config.CONFIG_DATALOADER[DK.D3_SIZE],
                                 example['X'].shape[-1]])

        print_number_of_zeros("before filling")
        #print(np.unique(example['X']))

        for patient_index, patient_path in tqdm(enumerate(self.data_storage.get_paths(storage_path=self.raw_path))):
            name = patient_path.split(self.config.CONFIG_PATHS[PK.SYS_DELIMITER])[-1].split('.')[0]
            condition = example["PatientName"] == name
            _data = self.data_storage.get_datas(data_path=patient_path)

            example_df = pd.DataFrame(example['indexes_in_datacube'], columns=['x', 'y'])
            example_df['index'] = example_df.index  # Keep original index to map back
            example_df['PatientName'] = example["PatientName"]  # Don't change the order x, y, index, PatientName

            data_df = pd.DataFrame(_data['indexes_in_datacube'], columns=['x', 'y'])
            data_df['index'] = data_df.index

            # Merge on coordinates
            merged_df = pd.merge(example_df[example_df['PatientName'] == name],
                                 data_df,
                                 on=['x', 'y'],
                                 suffixes=('_example', '_data'))

            # Update using numpy advanced indexing
            merged_df_numpy = merged_df.to_numpy()

            example['X'][merged_df_numpy[:, 2].astype(int)] = _data['X'][merged_df_numpy[:, -1].astype(int)]

        not_empty = print_number_of_zeros("after filling")
        example['X'] = example['X'][not_empty]
        #print(example['X'].shape)
        #print(np.unique(example['X']))

        np.savez(os.path.join(self.shuffle_saving_path, example_name+'.npz'), **example)
        adjust_meta_file()

    def check_piles_number(self):
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

    def __remove_files(self, extension):
        paths = glob(os.path.join(self.shuffle_saving_path, f"*{extension}*"))
        for p in paths:
            os.remove(p)

    def __create_piles(self, use_piles: List[int]):
        print("----Piles creating started----")
        print(f"Pile number: {self.piles_number}")
        if use_piles.__len__() < self.piles_number:
            print(f"Create only shuffle piles for the indexes: {use_piles}")

        # remove previous piles if they exist
        self.__remove_files(PILE_NAME)

        # create clear piles
        for i in use_piles:
            open(os.path.join(self.shuffle_saving_path, f"{i}{PILE_NAME}"), "w").close()  # creating of an empty file

        print("----Piles creating finished----")

    def __split_into_piles(self, use_piles: List[int]):
        print("--Splitting into piles started--")

        for pat_idx, p in tqdm(enumerate(self.data_storage.get_paths(storage_path=self.raw_path))):
            # clear piles for new randon numbers
            piles = self.__create_empty_piles(piles_length=len(use_piles))

            name = self.data_storage.get_name(path=p)
            _data = self.data_storage.get_datas(data_path=p)

            data = {n: a[...] for n, a in _data.items()}

            # get all labels in data archive
            unique_labels = np.unique(data[self.dict_names[1]])
            for label in unique_labels:
                # get all indexes on axis 0 per label
                indexes = np.where(data[self.dict_names[1]] == label)[0]
                # create a list with shuffle indexes (indexes between 0 and self.piles_number)
                sh_indexes = [idx % self.piles_number for idx in range(indexes.shape[0])]
                # shuffle the indexes
                random.shuffle(sh_indexes)
                sh_indexes = np.array(sh_indexes)
                # add all data indexes per pile that will be used
                for i, pile_idx in enumerate(use_piles):
                    piles[i] += list(indexes[sh_indexes == pile_idx])

            # fill random distribution to files
            """for it in range(data[self.dict_names[0]].shape[0]):
                pile = random.randint(0, self.piles_number - 1)
                if pile in use_piles:
                    piles[use_piles.index(pile)].append(it)"""

            # get array with patient index and name
            for i_pile, pile in zip(use_piles, piles):
                _names = [name] * len(pile)
                _indexes = [pat_idx] * len(pile)

                values = {}
                for k in data.keys():
                    if k in self.dict_names:
                        values[k] = data[k][pile]

                values[self.dict_names[2]] = _names
                values[self.dict_names[3]] = _indexes

                # save pile
                pickle.dump(values, open(os.path.join(self.shuffle_saving_path, str(i_pile) + PILE_NAME), 'ab'))

        print("--Splitting into piles finished--")

    def __shuffle_piles(self):
        print("----Shuffling of piles started----")
        piles_paths = glob(os.path.join(self.shuffle_saving_path, f"*{PILE_NAME}"))

        # load all piles and there sub dictionary's
        for pile in tqdm(piles_paths):
            data = []
            with open(pile, "rb") as fr:
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

            os.remove(pile)

            # save shuffled date and meta information
            sh_name = f"{SHUFFLE_GROUP_NAME}_{os.path.splitext(os.path.basename(pile))[0]}"
            write_meta_info(save_dir=self.shuffle_saving_path, file_name=sh_name,
                            labels=sh_data[self.dict_names[1]], names=sh_data[self.dict_names[2]],
                            names_idx=sh_data[self.dict_names[3]], X_shape=sh_data[self.dict_names[0]].shape,
                            typ=self.dataset_typ)

            self._save_file(file_name=sh_name, sh_data=sh_data)

        print("----Shuffling of piles finished----")

    def __all_pile_idx(self) -> List[int]:
        return [i for i in range(self.piles_number)]

    @staticmethod
    def __create_empty_piles(piles_length) -> List[list]:
        return [[] for _ in range(piles_length)]

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
