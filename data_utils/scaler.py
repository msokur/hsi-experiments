import abc

import numpy as np
import os
from tqdm import tqdm
from sklearn import preprocessing
import pickle

from util.compare_distributions import DistributionsChecker
from data_utils.data_archive import DataArchive
from configuration.parameter import (
    DICT_X, DICT_y, DICT_IDX,
    SCALER_FILE
)


class Scaler:
    def __init__(self, config, preprocessed_path, data_archive: DataArchive, scaler_file=None, scaler_path=None,
                 dict_names=None):
        self.config = config
        self.preprocessed_path = preprocessed_path
        self.scaler_path = scaler_path
        self.dict_names = dict_names
        self.data_archive = data_archive
        if self.dict_names is None:
            self.dict_names = [DICT_X, DICT_y, DICT_IDX]
        
        if self.scaler_path is not None: 
            self.scaler = Scaler.scaler_restore(self.scaler_path)
        else:
            X = self._get_data_for_fit()
            self.scaler = self._fit(X)
            if scaler_file is None:
                scaler_file = SCALER_FILE
            self.scaler_save(self.scaler, os.path.join(self.preprocessed_path, scaler_file))
            
    @abc.abstractmethod
    def _get_data_for_fit(self):
        pass
    
    @abc.abstractmethod
    def _fit(self, X):
        pass
    
    @abc.abstractmethod
    def _transform(self, X):
        pass
            
    def X_y_concatenate(self):
        paths = self.data_archive.get_paths(archive_path=self.preprocessed_path)

        X_s, y_s, indexes_s = self._get_shapes(paths[0])
        X, y, indexes = np.empty(shape=X_s), np.empty(shape=y_s), np.empty(shape=indexes_s)
        for data in tqdm(self.data_archive.all_data_generator(archive_path=self.preprocessed_path)):
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
                
            X = self._transform(X)

            # reshape back if 3d
            if _3d:
                X = np.reshape(X, shapes)

        return X

    def iterate_over_archives_and_save_scaled_X(self, destination_path):
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        for data in tqdm(self.data_archive.all_data_generator(archive_path=self.preprocessed_path)):
            X = data[self.dict_names[0]]
            X = self.scale_X(X)

            self.data_archive.save_data(save_path=self.data_archive.get_path(file=data),
                                        data_name=self.dict_names[0],
                                        data=X)

    def _get_shapes(self, path):
        datas = self.data_archive.get_datas(data_path=path)
        X, y, idx = datas[self.dict_names[0]].shape, datas[self.dict_names[1]].shape, datas[self.dict_names[2]].shape
        return [0, X[-1]], [0] if len(y) == 1 else [0, y[-1]], [0, idx[-1]]
            

class NormalizerScaler(Scaler):
    def __init__(self, *args, **kwargs):
        print('NormalizerScaler is created')
        super().__init__(*args, **kwargs)
        
    def _get_data_for_fit(self):
        return []
    
    def _fit(self, X):
        return preprocessing.Normalizer()
    
    def _transform(self, X):
        return self.scaler.fit_transform(X)


class StandardScalerTransposed(Scaler):
    def __init__(self, *args, **kwargs):
        print('StandardScalerTransposed is created')
        super().__init__(*args, **kwargs)
        
    def _get_data_for_fit(self):
        return []
    
    def _fit(self, X):
        return preprocessing.StandardScaler()
    
    def _transform(self, X):
        return self.scaler.fit_transform(X.T).T
    
    
    
        
    
