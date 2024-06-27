from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
from scipy.signal import savgol_filter
import numpy as np

from data_utils.smoothing import MedianFilter, GaussianFilter, SavGolFilter
from configuration.keys import DataLoaderKeys as DLK

np.random.seed(106)
spectrum = np.random.randn(20, 20, 100)



def test_MF_1d(test_config):
    MF = MedianFilter(test_config)
    _1d_smoothed_spectrum = MF.smooth_1d(spectrum)
    size = test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]

    assert np.all(_1d_smoothed_spectrum[0, 0] == median_filter(spectrum[0, 0], size=size))


def test_MF_2d(test_config):
    MF = MedianFilter(test_config)
    _2d_smoothed_spectrum = MF.smooth_2d(spectrum)
    size = test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]

    assert np.all(_2d_smoothed_spectrum[..., 0] == median_filter(spectrum[..., 0], size=size))


def test_MF_3d(test_config):
    MF = MedianFilter(test_config)
    _3d_smoothed_spectrum = MF.smooth_3d(spectrum)
    size = test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]

    assert np.all(_3d_smoothed_spectrum == median_filter(spectrum, size=size))


def test_gaussian_1d(test_config):
    gaussian = GaussianFilter(test_config)
    _1d_smoothed_spectrum = gaussian.smooth_1d(spectrum)
    size = test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]

    assert np.all(_1d_smoothed_spectrum[0, 0] == gaussian_filter1d(spectrum[0, 0], sigma=size))


def test_gaussian_2d(test_config):
    gaussian = GaussianFilter(test_config)
    _2d_smoothed_spectrum = gaussian.smooth_2d(spectrum)
    size = test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]

    assert np.all(_2d_smoothed_spectrum[..., 0] == gaussian_filter(spectrum[..., 0], sigma=size))


def test_gaussian_3d(test_config):
    gaussian = GaussianFilter(test_config)
    _3d_smoothed_spectrum = gaussian.smooth_3d(spectrum)
    size = test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]

    assert np.all(_3d_smoothed_spectrum == gaussian_filter(spectrum, sigma=size))


def test_savgol_1d(test_config):
    savgol_size = [9, 2]
    test_config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE] = [9, 2]
    savgol = SavGolFilter(test_config)
    _1d_smoothed_spectrum = savgol.smooth_1d(spectrum)

    assert np.all(_1d_smoothed_spectrum[0, 0] == savgol_filter(spectrum[0, 0],
                                                               window_length=savgol_size[0],
                                                               polyorder=savgol_size[1], axis=-1))
