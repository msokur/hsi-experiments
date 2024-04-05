from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
import abc
import numpy as np

from configuration.keys import DataLoaderKeys as DLK


class Smoother:
    def __init__(self, config):
        self.config = config
        self.size = self.config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]
    
    @abc.abstractmethod
    def smooth_1d(self, X):
        pass

    @abc.abstractmethod
    def smooth_2d(self, X):
        pass

    def smooth(self, X):
        smoothing_dimension = self.config[DLK.SMOOTHING][DLK.SMOOTHING_DIMENSIONS]
        if smoothing_dimension == '1d':
            self.smooth_1d(X)
        if smoothing_dimension == '2d':
            self.smooth_2d(X)

        raise NotImplementedError("There is no implementation for SMOOTHING_DIMENSIONS, look in Dataloader.json config")

    def smooth1d_from_2d_input(self, X_2d):
        original_shape = X_2d.shape

        X_1d = np.reshape(X_2d, [original_shape[0] * original_shape[1], original_shape[2]])
        X_smoothed = self.smooth_1d(X_1d)
        X_smoothed_2d = np.reshape(X_smoothed, original_shape)

        return X_smoothed_2d
            
    
class MedianFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def smooth_1d(self, X):
        return median_filter(X, size=(1, self.size))

    def smooth_2d(self, X):
        return median_filter(X, size=self.size)


class GaussianFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def smooth_1d(self, X):
        return gaussian_filter1d(X, sigma=self.size)

    def smooth_2d(self, X):
        return gaussian_filter(X, sigma=self.size)
