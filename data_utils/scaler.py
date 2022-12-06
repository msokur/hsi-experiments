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
    def __init__(self, preprocessed_path, scaler_file=None, scaler_path=None):
        self.preprocessed_path = preprocessed_path
        self.scaler_path = scaler_path
        
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

        X, y, indexes = [], [], []
        for path in paths:
            data = np.load(path)
            _X, _y, _i = data['X'], data['y'], data['indexes_in_datacube']

            # check if data 3D
            if len(np.array(_X).shape) > 2:
                _X = DistributionsChecker.get_centers(_X)

            X += list(_X)
            y += list(_y)
            indexes += list(_i)

        X, y, indexes = np.array(X), np.array(y), np.array(indexes)

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
            X = data['X']
            X = self.scale_X(X)
            data = {n: a for n, a in data.items()}
            data['X'] = X.copy()
            np.savez(os.path.join(destination_path, DataLoaderDyn.get_name_easy(path)), **data)
            

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
    
    
    
        
    
