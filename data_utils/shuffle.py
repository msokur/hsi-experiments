import os
import random
import pickle
import numpy as np

from glob import glob
from tqdm import tqdm

from configuration.copy_py_files import copy_files


class Shuffle:
    def __init__(self, raw_paths, dict_names: list, prepro_conf: dict, paths_conf: dict, augmented=False):
        self.raw_paths = raw_paths
        self.prepro = prepro_conf
        self.paths = paths_conf
        self.dict_names = dict_names
        self.piles_number = self.prepro["PILES_NUMBER"]
        self.shuffle_saving_path = self.paths["SHUFFLED_PATH"]
        self.augmented = augmented

    def shuffle(self):
        print('--------Shuffling started--------')
        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(self.shuffle_saving_path,
                   self.prepro["FILES_TO_COPY"],
                   self.paths["SYSTEM_PATHS_DELIMITER"])

        self.__create_piles()
        self.__shuffle_piles()
        print('--------Shuffling finished--------')

    # ------------------divide all samples into piles_number files------------------
    def __create_piles(self):
        print('----Piles creating started----')

        # remove previous piles if they exist
        piles_paths = glob(os.path.join(self.shuffle_saving_path, '*pile*'))
        for p in piles_paths:
            os.remove(p)

        # create clear piles
        piles = []
        for i in range(self.piles_number):
            piles.append([])
            open(os.path.join(self.shuffle_saving_path, str(i) + '.pile'), 'w').close()  # creating of an empty file

        print('--Splitting into piles started--')

        for i, p in tqdm(enumerate(self.raw_paths)):
            # clear piles for new randon numbers
            for pn in range(self.piles_number):
                piles[pn] = []

            name = p.split(self.paths["SYSTEM_PATHS_DELIMITER"])[-1].split(".")[0]
            _data = np.load(p)

            data = {n: a for n, a in _data.items()}
            X, y = data[self.dict_names[0]], data[self.dict_names[1]]

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

                values = [data[self.dict_names[0]][pile],
                          data[self.dict_names[1]][pile],
                          _names,
                          _indexes]
                if len(self.dict_names) > 4:
                    for nm in self.dict_names[4:]:
                        values.append(data[nm][pile])

                _values = {k: v for k, v in zip(self.dict_names, values)}
                pickle.dump(_values, open(os.path.join(self.shuffle_saving_path, str(i_pile) + '.pile'), 'ab'))

        print('--Splitting into piles finished--')

        print('----Piles creating finished----')

    def __shuffle_piles(self):
        print('----Shuffling of piles started----')
        piles_paths = glob(os.path.join(self.shuffle_saving_path, '*.pile'))
        print(len(piles_paths))

        for i, pp in tqdm(enumerate(piles_paths)):
            data = []
            with open(pp, 'rb') as fr:
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
            np.savez(os.path.join(self.shuffle_saving_path, 'shuffled' + str(i)),
                     **{n: a[indexes] for n, a in _data.items()})

        print('----Shuffling of piles finished----')
