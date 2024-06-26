from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
from scipy.signal import savgol_filter
import abc

from configuration.keys import DataLoaderKeys as DLK


class Smoother:
    def __init__(self, config):
        self.config = config
        self.size = self.config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]
        self.smoothing_dimension = self.config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_DIMENSIONS]

    @abc.abstractmethod
    def smooth_1d(self, spectrum):
        pass

    @abc.abstractmethod
    def smooth_2d(self, spectrum):
        pass

    @abc.abstractmethod
    def smooth_3d(self, spectrum):
        pass

    def print_type_of_smoother(self, smoother_type):
        print(f"Spectrum is smoothed with {smoother_type} filter! With dimensions: {self.smoothing_dimension};"
              f" and value - {self.size}")

    def smooth(self, spectrum):
        if self.smoothing_dimension == '1d':
            print("1d smoothing")
            return self.smooth_1d(spectrum)
        elif self.smoothing_dimension == '2d':
            print("2d smoothing")
            return self.smooth_2d(spectrum)
        elif self.smoothing_dimension == '3d':
            print("3d smoothing")
            return self.smooth_3d(spectrum)

        raise NotImplementedError(f"There is no implementation for SMOOTHING_DIMENSIONS {self.smoothing_dimension}, "
                                  "look in Dataloader.json config")


class MedianFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_type_of_smoother('Median filter')

    def smooth_1d(self, spectrum):
        return median_filter(spectrum, size=(1, 1, self.size))

    def smooth_2d(self, spectrum):
        return median_filter(spectrum, size=(self.size, self.size, 1))

    def smooth_3d(self, spectrum):
        return median_filter(spectrum, size=self.size)


class GaussianFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_type_of_smoother('Gaussian filter')

    def smooth_1d(self, spectrum):
        return gaussian_filter1d(spectrum, sigma=self.size)

    def smooth_2d(self, spectrum):
        return gaussian_filter(spectrum, sigma=self.size, axes=[0, 1])

    def smooth_3d(self, spectrum):
        return gaussian_filter(spectrum, sigma=self.size)


class SavGolFilter(Smoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_type_of_smoother('Savatsky-Golai filter')

    def smooth_1d(self, spectrum):
        return savgol_filter(x=spectrum, window_length=self.size[0], polyorder=self.size[1], axis=-1)

    def smooth_2d(self, spectrum):
        raise NotImplementedError("There is no implementation for 2d Savatsky-Golai filter yet")

    def smooth_3d(self, spectrum):
        raise NotImplementedError("There is no implementation for 3d Savatsky-Golai filter yet")
