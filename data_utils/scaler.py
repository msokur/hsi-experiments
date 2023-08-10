import abc
from typing import Tuple

import numpy as np
import os
from tqdm import tqdm
from glob import glob
from sklearn import preprocessing
import pickle
import math

from util.compare_distributions import DistributionsChecker
from data_utils.data_archive import DataArchive
from configuration.parameter import (
    DICT_X, DICT_y, DICT_IDX,
    SCALER_FILE
)


class Scaler:
    def __init__(self, preprocessed_path, data_archive: DataArchive, scaler_file=None, scaler_path=None,
                 dict_names=None):
        self.preprocessed_path = preprocessed_path
        self.scaler_path = scaler_path
        self.dict_names = dict_names
        self.data_archive = data_archive
        if self.dict_names is None:
            self.dict_names = [DICT_X, DICT_y, DICT_IDX]
        
        if self.scaler_path is not None: 
            self.scaler = Scaler.scaler_restore(self.scaler_path)
        else:
            X = self.get_data_for_fit()
            self.scaler = self.fit(X)
            if scaler_file is None:
                scaler_file = SCALER_FILE
            self.scaler_save(self.scaler, os.path.join(self.preprocessed_path, scaler_file))
            
    @abc.abstractmethod
    def get_data_for_fit(self):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass
    
    @abc.abstractmethod
    def transform(self, X):
        pass
            
    def X_y_concatenate(self):
        paths = self.data_archive.get_paths(archive_path=self.preprocessed_path)
        print(paths)
        X_s, y_s, indexes_s = self.get_shapes(paths[0])
        X, y, indexes = np.empty(shape=X_s), np.empty(shape=y_s), np.empty(shape=indexes_s)
        for data in tqdm(self.data_archive.all_data_generator()):
            _X, _y, _i = data[self.dict_names[0]], data[self.dict_names[1]], data[self.dict_names[2]]

            # check if data 3D
            if len(np.array(_X).shape) > 2:
                _X_1d = DistributionsChecker.get_centers(_X)
            else:
                _X_1d = _X

            X = np.concatenate((X, _X_1d), axis=0)
            y = np.concatenate((y, _y), axis=0)
            indexes = np.concatenate((indexes, _i), axis=0)
            del _X
            del _y
            del _i
            del _X_1d
            del data

        return X, y, indexes

    @staticmethod
    def scaler_save(scaler, scaler_path):
        pickle.dump(scaler, open(scaler_path, 'wb'))

    @staticmethod
    def scaler_restore(scaler_path):
        return pickle.load(open(scaler_path, 'rb'))

    def scale_X(self, X):
        _3d = False
        shapes = []

        if X.shape[0] != 0:  # reshape X if 3d
            if len(X.shape) > 2:
                _3d = True
                shapes = X.shape
                X = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[-1]))
                
            X = self.transform(X)

            # reshape back if 3d
            if _3d:
                X = np.reshape(X, shapes)

        return X

    def iterate_over_archives_and_save_scaled_X(self, destination_path):
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        for data in tqdm(self.data_archive.all_data_generator()):
            X = data[self.dict_names[0]]
            X = self.scale_X(X)

            datas = {n: a for n, a in data.items()}
            datas[self.dict_names[0]] = X.copy()
            self.data_archive.save_group(save_path=destination_path,
                                         group_name=os.path.split(os.path.abspath(data))[-1],
                                         datas=datas)

    def get_shapes(self, path):
        datas = self.data_archive.get_datas(data_path=path)
        X, y, idx = datas[self.dict_names[0]].shape, datas[self.dict_names[1]].shape, datas[self.dict_names[2]].shape
        return [0, X[-1]], [0] if len(y) == 1 else [0, y[-1]], [0, idx[-1]]
            

class NormalizerScaler(Scaler):
    def __init__(self, *args, **kwargs):
        print('NormalizerScaler is created')
        super().__init__(*args, **kwargs)
        
    def get_data_for_fit(self):
        return []
    
    def fit(self, X):
        return preprocessing.Normalizer()
    
    def transform(self, X):
        return self.scaler.fit_transform(X)


class SNV(Scaler):
    def __init__(self, *args, **kwargs):
        print('StandardScaler is created')
        super().__init__(*args, **kwargs)

    def get_data_for_fit(self):
        return []

    def fit(self, X):
        scaler = preprocessing.StandardScaler()
        samples, features, shape = self.get_samples_features_shape()
        mean = self.get_mean(samples=samples, features=features, shape=shape)
        var = self.get_var(mean=mean, samples=samples, features=features, shape=shape)
        std = self.get_std(var)
        scaler.n_samples_seen_ = samples
        scaler.n_features_in_ = features
        scaler.mean_ = mean
        scaler.var_ = var
        scaler.scale_ = std
        return scaler

    def transform(self, X):
        self.scaler.transform(X)

    def get_samples_features_shape(self) -> Tuple[np.int64, int, tuple]:
        samples = np.int64(0)
        features = 0
        shape = (0, 0)
        for data in tqdm(self.data_archive.all_data_generator()):
            shape = data["X"].shape
            features = shape[-1]
            samples += shape[0]

        return samples, features, shape

    def get_mean(self, samples: np.int64, features: int, shape: tuple) -> np.ndarray:
        mean_ = np.zeros(shape=features)
        for data in tqdm(self.data_archive.all_data_generator()):
            if len(shape) > 2:
                center = math.floor(shape[1] / 2)
                X = data["X"][:, center, center, ...]
            else:
                X = data["X"][...]

            mean_ += X.sum(axis=0)

        return mean_ / samples

    def get_var(self, mean: np.ndarray, samples: np.int64, features: int, shape: tuple) -> np.ndarray:
        var_ = np.zeros(shape=features)
        for data in tqdm(self.data_archive.all_data_generator()):
            if len(shape) > 2:
                center = math.floor(shape[1] / 2)
                X = data["X"][:, center, center, ...]
            else:
                X = data["X"][...]
            var_ += np.sum(a=(X - mean) ** 2, axis=0)

        return var_ / samples

    @staticmethod
    def get_std(var: np.ndarray) -> np.ndarray:
        return np.sqrt(var)


class StandardScaler(Scaler):
    def __init__(self, *args, **kwargs):
        print('StandardScaler is created')
        super().__init__(*args, **kwargs)
        
    def get_data_for_fit(self):
        X, _, _ = self.X_y_concatenate()
        return X
    
    def fit(self, X):
        return preprocessing.StandardScaler().fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)


class StandardScalerTransposed(Scaler):
    def __init__(self, *args, **kwargs):
        print('StandardScalerTransposed is created')
        super().__init__(*args, **kwargs)
        
    def get_data_for_fit(self):
        return []
    
    def fit(self, X):
        return preprocessing.StandardScaler()
    
    def transform(self, X):
        return self.scaler.fit_transform(X.T).T
    
    
    
        
    
