import abc

import numpy as np
import os
from tqdm import tqdm
from glob import glob
from sklearn import preprocessing
import pickle

from util.compare_distributions import DistributionsChecker
from data_utils.data_loaders.data_loader_dyn import DataLoaderDyn


class Scaler:
    def __init__(self, preprocessed_path, scaler_file=None, scaler_path=None, dict_names=None):
        self.preprocessed_path = preprocessed_path
        self.scaler_path = scaler_path
        if dict_names is None:
            self.dict_names = ["X", "y", "indexes_in_datacube"]
        
        if self.scaler_path is not None: 
            self.scaler = Scaler.scaler_restore(self.scaler_path)
        else:
            X = self.get_data_for_fit()
            self.scaler = self.fit(X)
            if scaler_file is None:
                scaler_file = "scaler.scaler"
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
        paths = glob(os.path.join(self.preprocessed_path, '*.npz'))

        X_s, y_s, indexes_s = self.get_shapes(paths[0])
        X, y, indexes = np.empty(shape=X_s), np.empty(shape=y_s), np.empty(shape=indexes_s)
        for path in paths:
            data = np.load(path)
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

    def iterate_over_archives_and_save_scaled_X(self, root_path, destination_path):
        paths = glob(os.path.join(root_path, '*.npz'))

        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        for path in tqdm(paths):
            data = np.load(path)
            X = data[self.dict_names[0]]
            X = self.scale_X(X)
            data = {n: a for n, a in data.items()}
            data[self.dict_names[0]] = X.copy()
            np.savez(os.path.join(destination_path, DataLoaderDyn().get_name(path)), **data)

    def get_shapes(self, path):
        datas = np.load(path)
        X, y, idx = datas[self.dict_names[0]].shape, datas[self.dict_names[1]].shape, datas[self.dict_names[2]].shape
        return [0, X[-1]], [0] if len(y) == 1 else [0, y[-1]], [0, idx[-1]]
            

class NormalizerScaler(Scaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_data_for_fit(self):
        return []
    
    def fit(self, X):
        return preprocessing.Normalizer()
    
    def transform(self, X):
        return self.scaler.fit_transform(X)
    

class StandardScaler(Scaler):
    def __init__(self, *args, **kwargs):
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
    
    
    
        
    
