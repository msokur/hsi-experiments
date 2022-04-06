import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, os.path.join(parentdir, 'utils')) 

import config

import os
import numpy as np
import random
import glob
from tqdm import tqdm
import pickle
from scipy.signal import savgol_filter
from sklearn import preprocessing

import provider
from data_loaders.data_loader_base import DataLoader


'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''


class Preprocessor():
    def __init__(self, load_name_for_x='X',
                 load_name_for_y='y',
                 load_name_for_name='PatientName',
                 piles_number=100,
                 weights_filename='.weights',
                 dict_names=['PatientIndex', 'indexes_in_datacube', 'weights']):
                 #dict_names=['PatientIndex', 'indexes_in_datacube']):
        self.load_name_for_name = load_name_for_name
        self.dict_names = [load_name_for_x, load_name_for_y, load_name_for_name]
        for name in dict_names:
            self.dict_names.append(name)
        self.piles_number = piles_number
        self.weights_filename = weights_filename

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

            name = p.split(config.SYSTEM_PATHS_DELIMITER)[-1].split(".")[0]
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

            indexes = np.arange(_data[self.dict_names[0]].shape[0])
            random.shuffle(indexes)

            os.remove(pp)
            np.savez(os.path.join(self.shuffle_saving_path, 'shuffled' + str(i)),
                     **{n: a[indexes] for n, a in _data.items()})

        print('----Shuffling of piles finished----')

    '''Add here some specific preprocessing if needed'''

    @staticmethod
    def preprocess(X, y):
        # ill_indexes = np.flatnonzero(_y == 1)   #TODO remove!
        # _X[ill_indexes, :-1] = savgol_filter(_X[ill_indexes, :-1], 5, 2) #TODO remove!

        # X[X < 0] = 0.
        # X = preprocessing.Normalizer().transform(X[:, :-1]) #TODO be careful

        # ill_indexes = np.flatnonzero(y == 1)

        # X = X[:, :-1]
        # X[X<0] = 0.
        # X[ill_indexes] = savgol_filter(X[ill_indexes], 7, 2)
        # X[X<0] = 0.
        # X = preprocessing.Normalizer().transform(X)
        # X = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

        return X, y

    def shuffle(self, paths, piles_number, shuffle_saving_path, augmented=False):
        print('--------Shuffling started--------')
        self.raw_paths = paths
        self.piles_number = piles_number
        self.shuffle_saving_path = shuffle_saving_path
        self.augmented = augmented

        if not os.path.exists(self.shuffle_saving_path):
            os.mkdir(self.shuffle_saving_path)

        self.__create_piles()
        self.__shuffle_piles()
        print('--------Shuffling finished--------')

    def X_y_concatenate(self, root_path):
        paths = glob.glob(os.path.join(root_path, '*.npz'))

        X, y, indexes = [], [], []
        for path in paths:
            data = np.load(path)
            _X, _y, _i = data['X'], data['y'], data['indexes_in_datacube']

            X += list(_X)
            y += list(_y)
            indexes += list(_i)

        X, y, indexes = np.array(X), np.array(y), np.array(indexes)

        return X, y, indexes

    def scaler_save(self, scaler, scaler_path):
        pickle.dump(scaler, open(scaler_path, 'wb'))

    def scaler_restore(self, scaler_path):
        return pickle.load(open(scaler_path, 'rb'))

    def fit_scale_from_path(self, root_path, scaler_saving_path, scaling_type=config.NORMALIZATION_TYPE):
        X, _, _ = self.X_y_concatenate(root_path)
        return self.scale_from_X(X, scaler_saving_path, scaling_type=scaling_type)

    def fit_scale_from_X(self, X, scaler_saving_path, scaling_type=config.NORMALIZATION_TYPE):
        scaler = None

        if scaling_type == 0:  # svn
            scaler = preprocessing.StandartScaler().fit(X)
        else:  # l2_norm
            scaler = preprocessing.Normalizer().fit(X)

        self.scaler_save(scaler, scaler_saving_path)
        return scaler.transform(X)

    def scale_X(self, X, scaler_path):
        _3d = False
        shapes = []

        if X.shape[0] != 0:  # reshape X if 3d
            if len(X.shape) > 2:
                _3d = True
                shapes = X.shape
                X = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[-1]))

            # restore scaler
            if config.NORMALIZATION_TYPE == 'svn_T':
                scaler = preprocessing.StandardScaler().fit(X.T)
            else:
                scaler = self.scaler_restore(scaler_path)

            # transform X
            if config.NORMALIZATION_TYPE == 'svn_T':
                X = scaler.transform(X.T).T
            else:
                X = scaler.transform(X)

            # reshape back if 3d
            if _3d:
                X = np.reshape(X, shapes)

        return X

    def scaledData_save(self, root_path, destination_path, scaler_path):
        paths = glob.glob(os.path.join(root_path, '*.npz'))

        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        for path in tqdm(paths):
            data = np.load(path)
            X = data['X']
            X = self.scale_X(X, scaler_path)
            data = {n: a for n, a in data.items()}
            data['X'] = X.copy()
            np.savez(os.path.join(destination_path, DataLoader.get_name_easy(path)), **data)

    def __split_arrays(self, *args):
        # ---------------splitting into archives----------
        chunks = args[0].shape[0] // self.batch_size
        chunks_max = chunks * self.batch_size

        arrs = [np.array_split(arg[:chunks_max], chunks) for arg in args]

        # ---------------saving of the non equal last part for the future partition---------
        for i in range(len(args)):
            self.rest_arrs[i] += list(args[i][chunks_max:])

        # ---------------saving of the non equal last part for the future partition---------
        ind = len(glob.glob(os.path.join(self.archives_of_batch_size_saving_path, "*")))
        for row in range(chunks):
            arch = {}
            for i, n in enumerate(self.dict_names):
                arch[n] = arrs[i][row]

            if config.WITH_PREPROCESS_DURING_SPLITTING:
                _X, _y = arrs[0][row], arrs[1][row]
                _X, _y = Preprocessor.preprocess(_X, _y)
                arch[self.dict_names[0]] = _X.copy()
                arch[self.dict_names[1]] = _y.copy()

            np.savez(os.path.join(self.archives_of_batch_size_saving_path, 'batch' + str(ind)), **arch)
            ind += 1

    def __init_rest(self):
        self.rest_arrs = []
        for i in range(len(self.dict_names)):
            self.rest_arrs.append([])

    def __check_dict_names(self, dict_names, data_):
        diff_dict_names = set(self.dict_names) - set(data_)
        diff_dataset = set(data_) - set(self.dict_names)

        if len(diff_dict_names):
            print(f'WARNING! dict_names {diff_dict_names} are not in dataset')
        if len(diff_dataset):
            print(f'WARNING! dict_names {diff_dataset} are not in dict_names of Preprocessor')

    def split_data_into_npz_of_batch_size(self, paths,
                                          batch_size,
                                          archives_of_batch_size_saving_path,
                                          scaler_saving_path,
                                          except_names=[]):

        print('--------Splitting into npz of batch size started--------')
        self.batch_size = batch_size
        self.archives_of_batch_size_saving_path = archives_of_batch_size_saving_path

        if not os.path.exists(archives_of_batch_size_saving_path):
            os.mkdir(archives_of_batch_size_saving_path)

        for except_name in except_names:
            print(f'We except {except_name}')

        # ------------removing of previously generated archives (of the previous CV step) ----------------
        files = glob.glob(os.path.join(archives_of_batch_size_saving_path, '*.npz'))
        for f in files:
            os.remove(f)

        self.__init_rest()

        for p in tqdm(paths):
            # ------------ except_indexes filtering ---------------
            data_ = np.load(p)
            self.__check_dict_names(self.dict_names, data_)
            self.dict_names = list(set(self.dict_names).intersection(set(data_)))

            data = {name: data_[name] for name in self.dict_names}
            p_names = data[self.load_name_for_name]

            for except_name in except_names:
                indexes = np.flatnonzero(p_names != except_name)
                if indexes.shape[0] == 0:
                    print(f'WARNING! For except_name {except_name} no except_samples were found')
                
                p_names = p_names[indexes]
                data = {n: a[indexes] for n, a in data.items()}

            indexes = np.zeros(data['y'].shape).astype(bool)
            for label in config.LABELS_OF_CLASSES_TO_TRAIN:
                indexes = indexes | (data['y'] == label)

            indexes = np.flatnonzero(indexes)
            data = {n: a[indexes] for n, a in data.items()}

            self.__split_arrays(*[a for _, a in data.items()])

        # ------------------save rest of rest archives----------------
        rest_arrs = [np.array(rest_arr) for rest_arr in self.rest_arrs]
        self.__init_rest()

        if rest_arrs[0].shape[0] >= batch_size:
            self.__split_arrays(*rest_arrs)

        print('--------Splitting into npz of batch size finished--------')

    def weightedData_save(self, root_path, weights):
        paths = glob.glob(os.path.join(root_path, '*.npz'))
        for i, path in tqdm(enumerate(paths)):
            data = np.load(path)
            X, y = data['X'], data['y']
            weights_ = np.zeros(y.shape)

            for j in np.unique(y):
                weights_[y == j] = weights[i, j]

            data = {n: a for n, a in data.items()}
            data['weights'] = weights_

            np.savez(os.path.join(root_path, DataLoader.get_name_easy(path)), **data)

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
        y_unique = pickle.load(open(os.path.join(root_path, DataLoader.get_labels_filename()), 'rb'))

        quantities = []
        for path in tqdm(paths):
            data = np.load(path)
            X, y = data['X'], data['y']

            quantity = []
            for y_u in y_unique:
                quantity.append(X[y == y_u].shape[0])

            quantities.append(quantity)

        quantities = np.array(quantities)

        summ = np.sum(quantities[:, config.LABELS_OF_CLASSES_TO_TRAIN])
        weights = summ / quantities

        data = {
            'weights': weights,
            'summ': summ,
            'quantities': quantities
        }

        with open(weights_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return weights

    @staticmethod
    def get_execution_flags_for_pipeline_with_all_true():
        return {
            "load_data_with_dataloader": True,
            "add_sample_weights": True,
            "scale": True,
            "shuffle": True
        }

    def pipeline(self, root_path,
                 preprocessed_path,
                 scaler_path=None,
                 scaler_name='scaler',
                 execution_flags=None):
        if execution_flags is None:
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()

        if not os.path.exists(preprocessed_path):
            os.mkdir(preprocessed_path)

        # ---------Data reading part--------------
        if execution_flags['load_data_with_dataloader']:
            dataLoader = provider.get_data_loader()
            dataLoader.files_read_and_save_to_npz(root_path, preprocessed_path)

        # ----------weights part------------------
        if execution_flags['add_sample_weights']:
            weights = self.weights_get_or_save(preprocessed_path)
            self.weightedData_save(preprocessed_path, weights)

        # ----------scaler part ------------------
        if execution_flags['scale']:
            if scaler_path is None and config.NORMALIZATION_TYPE != 'svn_T':
                scaler_path = os.path.join(preprocessed_path, scaler_name + config.SCALER_FILE_NAME)
                self.fit_scale_from_path(preprocessed_path, scaler_path)
            self.scaledData_save(preprocessed_path, preprocessed_path, scaler_path)

        # ----------shuffle part------------------
        if execution_flags['shuffle']:
            paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
            self.shuffle(paths,
                         self.piles_number,
                         os.path.join(preprocessed_path, 'shuffled'),
                         augmented=False)


if __name__ == '__main__':
    execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
    execution_flags['load_data_with_dataloader'] = True
    execution_flags['scale'] = False
    execution_flags['add_sample_weights'] = True
    execution_flags['shuffle'] = True

    for db in ['ColonData']:
    #for db in ['Colon_MedianFilter', 'Colon_SNV']:
    #for db in ['EsophagusDatabase', 'Esophagus_MedFilter', 'Esophagus_SNV']:
    #for db in ['DatabaseBrainMM', 'DatabaseBrainSNV', 'DatabaseBrainFMed']:

        preprocessor = Preprocessor()
        pth = os.path.join('/work/users/mi186veva/data_bea_db', db)
        #pth = os.path.join('C:\\Users\\tkachenko\\Desktop\\HSI\\bea\\databases', db, db)
        preprocessor.pipeline(pth,
                              os.path.join(pth, 'raw_3d_weighted'), execution_flags=execution_flags)
    #preprocessor.pipeline('../data_preprocessed/EsophagusDatabase',
    #                      '../data_preprocessed/EsophagusDatabase/raw_3d_weights', execution_flags=execution_flags)
    #paths = glob.glob('/work/users/mi186veva/data_bea/ColonData/raw_3d_weights/shuffled/*.npz')
    # print(len(paths))
    # preprocessor.shuffle(['/work/users/mi186veva/data_preprocessed/augmented/2019_07_12_11_15_49_.npz', '/work/users/mi186veva/data_preprocessed/augmented/2020_03_27_16_56_41_.npz'], 100, '/work/users/mi186veva/data_preprocessed/augmented_l2_norm/shuffled')
    # preprocessor.shuffle(paths, 100, '/work/users/mi186veva/data_preprocessed/raw_3d/shuffled', augmented=False)
    # preprocessor.shuffle(paths, 100, '/work/users/mi186veva/data_preprocessed/augmented/shuffled', augmented=True)

    # paths = glob.glob('/work/users/mi186veva/data_preprocessed/combi_with_raw_ill/shuffled/*.npz')
    #preprocessor.split_data_into_npz_of_batch_size(paths, config.BATCH_SIZE,
    #                                               '/work/users/mi186veva/data_bea/ColonData/raw_3d_weights/batch_sized',
    #                                               "", except_names=['CP1'])

    # preprocessor.scale_from_path('/work/users/mi186veva/data_preprocessed/raw', '/work/users/mi186veva/data_preprocessed/raw/raw_all.scaler')
    # start = time.time()
    # preprocessor.scaledData_save('/work/users/mi186veva/data_preprocessed/raw', '/work/users/mi186veva/data_preprocessed/raw', '/work/users/mi186veva/data_preprocessed/raw/raw_all.scaler')
    # end = time.time()
    # print('Time of scaledData_save', end - start)
