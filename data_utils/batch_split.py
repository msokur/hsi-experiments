import os
import numpy as np

from glob import glob
from tqdm import tqdm


class BatchSplit:
    def __init__(self, labels_to_train: list, dict_names: list, batch_size: int):
        self.labels_to_train = labels_to_train
        self.dict_names = dict_names
        self.batch_size = batch_size

    def split_data_into_npz_of_batch_size(self, shuffled_paths,
                                          archives_of_batch_size_saving_path,
                                          except_names=None,
                                          valid_except_names=None):
        if valid_except_names is None:
            valid_except_names = []
        if except_names is None:
            except_names = []
        print('--------Splitting into npz of batch size started--------')
        valid_archives_saving_path = os.path.join(archives_of_batch_size_saving_path, 'valid')
        valid_except_names = list(valid_except_names)

        if not os.path.exists(archives_of_batch_size_saving_path):
            os.mkdir(archives_of_batch_size_saving_path)

        if not os.path.exists(valid_archives_saving_path):
            os.mkdir(valid_archives_saving_path)

        for except_name in except_names:
            print(f'We except {except_name}')

        for except_name in valid_except_names:
            print(f'We except VALID {except_name}')

        except_names += valid_except_names

        # ------------removing of previously generated archives (of the previous CV step) ----------------
        files = glob(os.path.join(archives_of_batch_size_saving_path, '*.npz'))
        for f in files:
            os.remove(f)

        files = glob(os.path.join(valid_archives_saving_path, '*.npz'))
        for f in files:
            os.remove(f)

        arch_rest = self.__init_rest()
        valid_rest = self.__init_rest()

        for p in tqdm(shuffled_paths):
            # ------------ except_indexes filtering ---------------
            valid_data = {}
            for k in self.dict_names:
                valid_data[k] = []

            data_ = np.load(p)
            self.__check_dict_names(self.dict_names, data_)
            self.dict_names = list(set(self.dict_names).intersection(set(data_)))

            data = {name: data_[name] for name in self.dict_names}
            p_names = data[self.dict_names[2]]

            # ------------ get only needed classes--------------
            indexes = np.zeros(data['y'].shape).astype(bool)
            for label in self.labels_to_train:
                indexes = indexes | (data['y'] == label)

            indexes = np.flatnonzero(indexes)
            data = {n: a[indexes] for n, a in data.items()}
            p_names = p_names[indexes]

            # ------------ get validation data --------------
            valid_indexes = np.isin(p_names, valid_except_names)

            for k in self.dict_names:
                valid_data[k] += list(data[k][valid_indexes])

            # ------------ get data without excepted_names--------------
            for except_name in except_names:
                indexes = np.flatnonzero(p_names != except_name)
                if indexes.shape[0] == 0:
                    print(f'WARNING! For except_name {except_name} no except_samples were found')
                    # TODO if there more than 1 names, only the last one will filtered out

                p_names = p_names[indexes]
                data = {n: a[indexes] for n, a in data.items()}

            arch_rest_temp = self.__split_arrays(archives_of_batch_size_saving_path, data)
            valid_rest_temp = self.__split_arrays(valid_archives_saving_path, valid_data)

        # ---------------- save rest from archive an valid data ------------------
            for name in self.dict_names:
                arch_rest[name] += list(arch_rest_temp[name])
                valid_rest[name] += list(valid_rest_temp[name])

        if len(arch_rest[self.dict_names[0]]) >= self.batch_size:
            self.__split_arrays(archives_of_batch_size_saving_path, arch_rest)

        if len(valid_rest[self.dict_names[0]]) >= self.batch_size:
            self.__split_arrays(valid_archives_saving_path, valid_rest)

        print('--------Splitting into npz of batch size finished--------')

    def __split_arrays(self, path, data):
        # ---------------splitting into archives----------
        chunks = np.array(data[self.dict_names[0]]).shape[0] // self.batch_size
        chunks_max = chunks * self.batch_size

        if chunks > 0:
            data_ = {k: np.array_split(a[:chunks_max], chunks) for k, a in data.items()}

            ind = len(glob(os.path.join(path, "*")))
            for row in range(chunks):
                arch = {}
                for i, n in enumerate(self.dict_names):
                    arch[n] = data_[n][row]

                np.savez(os.path.join(path, 'batch' + str(ind)), **arch)
                ind += 1

        # ---------------saving of the non equal last part for the future partition---------
        rest = {k: a[chunks_max:] for k, a in data.items()}
        # ---------------saving of the non equal last part for the future partition---------
        return rest

    def __init_rest(self):
        rest_array = {}
        for name in self.dict_names:
            rest_array[name] = []
        return rest_array

    @staticmethod
    def __check_dict_names(dict_names, data_):
        diff_dict_names = set(dict_names) - set(data_)
        diff_dataset = set(data_) - set(dict_names)

        if len(diff_dict_names):
            print(f'WARNING! dict_names {diff_dict_names} are not in dataset')
        if len(diff_dataset):
            print(f'WARNING! dict_names {diff_dataset} are not in dict_names of Preprocessor')
