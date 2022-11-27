from scipy.ndimage import gaussian_filter, median_filter
import abc
from tqdm import tqdm
from glob import glob
import os
import numpy as np

# import config


class Smoother:
    def __init__(self, path, size):
        self.path = path
        self.size = size
    
    @abc.abstractmethod
    def smooth_func(self, X):
        pass
    
    def smooth(self):
        paths = glob(os.path.join(self.path, '*.npz'))
        
        for path in tqdm(paths):
            data = np.load(path)
            X = self.smooth_func(data['X'])
            data_ = {n: a for n, a in data.items()}
            data_['X'] = X.copy()
            np.savez(path, **data_)
            
    
class MedianFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def smooth_func(self, X):
        return median_filter(X, size=self.size)


class GaussianFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def smooth_func(self, X):
        return gaussian_filter(X, self.size)
