import abc
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from sklearn import preprocessing
import pickle

import config
from util.compare_distributions import DistributionsChecker
from data_loaders.data_loader_base import DataLoader

class Scaler:
    
    def __init__(self, preprocessed_path, scaler_path=None):
        self.preprocessed_path = preprocessed_path
        self.scaler_path = scaler_path
        
        if self.scaler_path is not None: 
            self.scaler = self.restore_scaler(self.scaler_path)
        else:
            X = self.get_data_for_fit()
            self.scaler = self.fit(X)
            self.scaler_save(self.scaler, os.path.join(self.preprocessed_path, config.SCALER_FILE_NAME+config.SCALER_FILE_EXTENTION))
            
    @abc.abstractmethod
    def get_data_for_fit(self):
        return
    
    @abc.abstractmethod
    def fit(self, X):
        return
    
    @abc.abstractmethod
    def transform(self, X):
        return
            
    def X_y_concatenate(self, _3D=config._3D):
        paths = glob(os.path.join(self.preprocessed_path, '*.npz'))

        X, y, indexes = [], [], []
        for path in paths:
            data = np.load(path)
            _X, _y, _i = data['X'], data['y'], data['indexes_in_datacube']
            
            if _3D:
                _X = DistributionsChecker.get_centers(_X)

            X += list(_X)
            y += list(_y)
            indexes += list(_i)

        X, y, indexes = np.array(X), np.array(y), np.array(indexes)

        return X, y, indexes

    def scaler_save(self, scaler, scaler_path):
        pickle.dump(scaler, open(scaler_path, 'wb'))

    def scaler_restore(self, scaler_path):
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
            np.savez(os.path.join(destination_path, DataLoader.get_name_easy(path)), **data)
            
            
    
    
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
    
    
    
        
    
