# import os
# import sys
# import inspect
import os
import numpy as np
import random
import glob
from tqdm import tqdm
import pickle
from shutil import copyfile
from configuration import get_config as conf

'''currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(1, os.path.join(parentdir, 'utils'))
'''
# import utils
import provider_dyn
from configuration.copy_py_files import copy_files

'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''


class Preprocessor:
    def __init__(self, prepro_dict: dict, path_dict: dict, loader_dict: dict):
        self.prepro = prepro_dict
        self.paths = path_dict
        self.loader = loader_dict
        self.dataloader = provider_dyn.get_data_loader(typ=self.loader["TYPE"],
                                                       loader_config=self.loader,
                                                       path_conf=self.paths)
        self.Scaler = None
        self.valid_archives_saving_path = None
        self.archives_of_batch_size_saving_path = None
        self.batch_size = None
        self.augmented = None
        self.shuffle_saving_path = None
        self.raw_paths = None
        self.load_name_for_name = self.prepro["DICT_NAMES"][2]
        self.load_name_for_X = self.prepro["DICT_NAMES"][0]
        self.load_name_for_y = self.prepro["DICT_NAMES"][1]
        self.dict_names = self.prepro["DICT_NAMES"]
        self.piles_number = self.prepro["PILES_NUMBER"]
        self.weights_filename = self.prepro["WEIGHT_FILENAME"]

    # ------------------divide all samples into piles_number files------------------
    def __create_piles(self):
        print('----Piles creating started----')

        # remove previous piles if they exist
        piles_paths = glob.glob(os.path.join(self.shuffle_saving_path, '*pile*'))
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
        piles_paths = glob.glob(os.path.join(self.shuffle_saving_path, '*.pile'))
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

    def shuffle(self, paths, piles_number, shuffle_saving_path, augmented=False):
        print('--------Shuffling started--------')
        self.raw_paths = paths
        self.piles_number = piles_number
        self.shuffle_saving_path = shuffle_saving_path
        self.augmented = augmented

        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        copy_files(shuffle_saving_path,
                   self.prepro["FILES_TO_COPY"],
                   self.paths["SYSTEM_PATHS_DELIMITER"])

        self.__create_piles()
        self.__shuffle_piles()
        print('--------Shuffling finished--------')

    def __split_arrays(self, path, data):
        # ---------------splitting into archives----------
        chunks = np.array(data[self.load_name_for_X]).shape[0] // self.batch_size
        chunks_max = chunks * self.batch_size

        data_ = {k: np.array_split(a[:chunks_max], chunks) for k, a in data.items()}

        # arrs = [np.array_split(arg[:chunks_max], chunks) for arg in args]

        # ---------------saving of the non equal last part for the future partition---------
        # for i in range(len(args)):
        #    self.rest_arrs[i] += list(args[i][chunks_max:])

        # ---------------saving of the non equal last part for the future partition---------
        ind = len(glob.glob(os.path.join(path, "*")))
        for row in range(chunks):
            arch = {}
            for i, n in enumerate(self.dict_names):
                arch[n] = data_[n][row]

            np.savez(os.path.join(path, 'batch' + str(ind)), **arch)
            ind += 1

    def __init_rest(self):
        self.rest_arrs = []
        for i in range(len(self.dict_names)):
            self.rest_arrs.append([])

    def __check_dict_names(self, dict_names, data_):
        diff_dict_names = set(dict_names) - set(data_)
        diff_dataset = set(data_) - set(self.dict_names)

        if len(diff_dict_names):
            print(f'WARNING! dict_names {diff_dict_names} are not in dataset')
        if len(diff_dataset):
            print(f'WARNING! dict_names {diff_dataset} are not in dict_names of Preprocessor')

    def split_data_into_npz_of_batch_size(self, paths,
                                          batch_size,
                                          archives_of_batch_size_saving_path,
                                          except_names=None,
                                          valid_except_names=None):
        if valid_except_names is None:
            valid_except_names = []
        if except_names is None:
            except_names = []
        print('--------Splitting into npz of batch size started--------')
        self.batch_size = batch_size
        self.archives_of_batch_size_saving_path = archives_of_batch_size_saving_path
        self.valid_archives_saving_path = os.path.join(self.archives_of_batch_size_saving_path, 'valid')
        valid_except_names = list(valid_except_names)

        if not os.path.exists(archives_of_batch_size_saving_path):
            os.mkdir(archives_of_batch_size_saving_path)

        if not os.path.exists(self.valid_archives_saving_path):
            os.mkdir(self.valid_archives_saving_path)

        copy_files(archives_of_batch_size_saving_path,
                   self.prepro["FILES_TO_COPY"],
                   self.paths["SYSTEM_PATHS_DELIMITER"])
        copy_files(self.valid_archives_saving_path,
                   self.prepro["FILES_TO_COPY"],
                   self.paths["SYSTEM_PATHS_DELIMITER"])

        for except_name in except_names:
            print(f'We except {except_name}')

        for except_name in valid_except_names:
            print(f'We except VALID {except_name}')

        except_names += valid_except_names

        # ------------removing of previously generated archives (of the previous CV step) ----------------
        files = glob.glob(os.path.join(archives_of_batch_size_saving_path, '*.npz'))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(self.valid_archives_saving_path, '*.npz'))
        for f in files:
            os.remove(f)

        self.__init_rest()

        # valid_data = {}
        # for k in self.dict_names:
        #    valid_data[k] = []

        for p in tqdm(paths):
            # ------------ except_indexes filtering ---------------
            valid_data = {}
            for k in self.dict_names:
                valid_data[k] = []

            data_ = np.load(p)
            self.__check_dict_names(self.dict_names, data_)
            self.dict_names = list(set(self.dict_names).intersection(set(data_)))

            data = {name: data_[name] for name in self.dict_names}
            p_names = data[self.load_name_for_name]

            # ------------ get only needed classes--------------
            indexes = np.zeros(data['y'].shape).astype(bool)
            for label in self.loader["LABELS_TO_TRAIN"]:
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

            self.__split_arrays(self.archives_of_batch_size_saving_path, data)  # *[a for _, a in data.items()])
            self.__split_arrays(self.valid_archives_saving_path, valid_data)  # *[a for _, a in valid_data.items()])

        print('--------Splitting into npz of batch size finished--------')

    def weights_get_from_file(self, root_path):
        weights_path = os.path.join(root_path, self.weights_filename)
        if os.path.isfile(weights_path):
            weights = pickle.load(open(weights_path, 'rb'))
            return weights['weights']
        else:
            raise ValueError("No .weights file was found in the directory, check given path")

    def weights_get_or_save(self, root_path):
        weights_path = os.path.join(root_path, self.weights_filename)

        paths = glob.glob(os.path.join(root_path, '*.npz'))
        y_unique = pickle.load(open(os.path.join(root_path, self.dataloader.get_labels_filename()), 'rb'))

        quantities = []
        for path in tqdm(paths):
            data = np.load(path)
            X, y = data['X'], data['y']

            quantity = []
            for y_u in y_unique:
                quantity.append(X[y == y_u].shape[0])

            quantities.append(quantity)

        quantities = np.array(quantities)

        sum_ = np.sum(quantities[:, self.loader["LABELS_TO_TRAIN"]])
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = sum_ / quantities

        weights[np.isinf(weights)] = 0

        data = {
            'weights': weights,
            'sum': sum_,
            'quantities': quantities
        }

        with open(weights_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return weights

    def weightedData_save(self, root_path, weights):
        paths = glob.glob(os.path.join(root_path, "*.npz"))
        for i, path in tqdm(enumerate(paths)):
            data = np.load(path)
            X, y = data["X"], data["y"]
            weights_ = np.zeros(y.shape)

            for j in np.unique(y):
                weights_[y == j] = weights[i, j]

            data_ = {n: a for n, a in data.items()}
            data_["weights"] = weights_

            np.savez(os.path.join(root_path, self.dataloader.get_name(path)), **data_)

    @staticmethod
    def get_execution_flags_for_pipeline_with_all_true():
        return {
            "load_data_with_dataloader": True,
            "add_sample_weights": True,
            "scale": True,
            "shuffle": True
        }

    def pipeline(self, execution_flags=None):
        if execution_flags is None:
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()

        root_path = self.paths["RAW_SOURCE_PATH"]
        preprocessed_path = self.paths["RAW_NPZ_PATH"]

        if not os.path.exists(preprocessed_path):
            os.makedirs(preprocessed_path)

        print('ROOT PATH', self.paths["RAW_SOURCE_PATH"])
        print('PREPROCESSED PATH', preprocessed_path)

        copy_files(preprocessed_path,
                   self.prepro["FILES_TO_COPY"],
                   self.paths["SYSTEM_PATHS_DELIMITER"])

        # ---------Data reading part--------------
        if execution_flags['load_data_with_dataloader']:
            self.dataloader.files_read_and_save_to_npz(root_path, preprocessed_path)

        # ----------weights part------------------
        if execution_flags['add_sample_weights']:
            weights = self.weights_get_or_save(preprocessed_path)
            self.weightedData_save(preprocessed_path, weights)

        # ----------scaler part ------------------
        if execution_flags['scale'] and self.prepro["NORMALIZATION_TYPE"] is not None:
            print('SCALER TYPE', self.prepro["NORMALIZATION_TYPE"])
            self.Scaler = provider_dyn.get_scaler(typ=self.prepro["NORMALIZATION_TYPE"],
                                                  preprocessed_path=preprocessed_path,
                                                  scaler_file=self.prepro["SCALER_FILE"],
                                                  scaler_path=self.prepro["SCALER_PATH"])
            self.Scaler.iterate_over_archives_and_save_scaled_X(preprocessed_path, preprocessed_path)

        # ----------shuffle part------------------
        if execution_flags['shuffle']:
            paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
            self.shuffle(paths,
                         self.piles_number,
                         self.paths["SHUFFLED_PATH"],
                         augmented=False)


if __name__ == '__main__':
    execution_flags_ = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
    execution_flags_['load_data_with_dataloader'] = True
    execution_flags_['add_sample_weights'] = True
    execution_flags_['scale'] = True
    execution_flags_['shuffle'] = True

    try:
        preprocessor = Preprocessor(prepro_dict=conf.PREPRO, path_dict=conf.PATHS, loader_dict=conf.DATALOADER)
        preprocessor.pipeline(execution_flags=execution_flags_)

        conf.telegram.send_tg_message("operations in preprocessor.py are successfully completed!")

    except Exception as e:
        conf.telegram.send_tg_message(f"ERROR! in Preprocessor {e}")

        raise e
