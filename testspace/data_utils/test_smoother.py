from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
from scipy.signal import savgol_filter
import numpy as np

import configuration.get_config as config
from data_utils.smoothing import MedianFilter, GaussianFilter, SavGolFilter
from configuration.keys import DataLoaderKeys as DLK


spectrum = np.random.randn(20, 20, 100)
size = config.CONFIG_DATALOADER[DLK.SMOOTHING][DLK.SMOOTHING_VALUE]


def test_MF_1d():
    MF = MedianFilter(config)
    _1d_smoothed_spectrum = MF.smooth_1d(spectrum)

    assert np.all(_1d_smoothed_spectrum[0, 0] == median_filter(spectrum[0, 0], size=size))


def test_MF_2d():
    MF = MedianFilter(config)
    _2d_smoothed_spectrum = MF.smooth_2d(spectrum)

    assert np.all(_2d_smoothed_spectrum[..., 0] == median_filter(spectrum[..., 0], size=size))


def test_MF_3d():
    MF = MedianFilter(config)
    _3d_smoothed_spectrum = MF.smooth_3d(spectrum)

    assert np.all(_3d_smoothed_spectrum == median_filter(spectrum, size=size))


def test_gaussian_1d():
    gaussian = GaussianFilter(config)
    _1d_smoothed_spectrum = gaussian.smooth_1d(spectrum)

    assert np.all(_1d_smoothed_spectrum[0, 0] == gaussian_filter1d(spectrum[0, 0], sigma=size))


def test_gaussian_2d():
    gaussian = GaussianFilter(config)
    _2d_smoothed_spectrum = gaussian.smooth_2d(spectrum)

    assert np.all(_2d_smoothed_spectrum[..., 0] == gaussian_filter(spectrum[..., 0], sigma=size))


def test_gaussian_3d():
    gaussian = GaussianFilter(config)
    _3d_smoothed_spectrum = gaussian.smooth_3d(spectrum)

    assert np.all(_3d_smoothed_spectrum == gaussian_filter(spectrum, sigma=size))


def test_savgol_1d():
    gaussian = SavGolFilter(config)
    savgol_size = [9, 2]
    _1d_smoothed_spectrum = gaussian.smooth_1d(spectrum)

    assert np.all(_1d_smoothed_spectrum[0, 0] == savgol_filter(spectrum[0, 0],
                                                               window_length=savgol_size[0],
                                                               polyorder=savgol_size[1]))
