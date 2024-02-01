import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve1d


def read_cube_dat(filename):
    data = np.fromfile(filename, dtype='>f')  # returns 1D array and reads file in big-endian binary format
    data_cube = data[3:].reshape(640, 480, 100)  # reshape to data cube and ignore first 3 values which are wrong
    return data_cube


def create_conv_fkt(width):
    gaussFkt = np.zeros((width * 3 + 1))
    sub1 = np.square((1/np.sqrt(np.log(2))) * width * 0.5)
    for i in range((width * 3 + 1)):
        gaussFkt[i] = np.exp(-np.square(i - (width * 3 / 2)) / sub1)
    return gaussFkt


def calc_rgb(cube, LUT_gamma, minWL=500, maxWL=995, WLsteps=5):

    RGB_image = np.zeros((cube.shape[0], cube.shape[1], 3), dtype=np.float)

    # for blue pixel take the 530-560nm
    blue_range_start = int((530 - minWL) / WLsteps)
    blue_range_end = int((560 - minWL) / WLsteps)
    RGB_image[:, :, 2] = cube[:, :, blue_range_start:blue_range_end].mean(axis=2)

    # for the green pixel take 540-590nm
    green_range_start = int((540 - minWL) / WLsteps)
    green_range_end = int((590 - minWL) / WLsteps)
    RGB_image[:, :, 1] = cube[:, :, green_range_start:green_range_end].mean(axis=2)

    # for the red pixel take 585 - 725 nm
    red_range_start = int((585 - minWL) / WLsteps)
    red_range_end = int((725 - minWL) / WLsteps)
    RGB_image[:, :, 0] = cube[:, :, red_range_start:red_range_end].mean(axis=2)

    # scale to 255
    factor = 1.02 * 255 * 1.5   # like in RGB-Image 1.5.vi
    RGB_image = np.clip((RGB_image * factor), 0, 255).astype(np.uint8)

    cv2.LUT(RGB_image, LUT_gamma, dst=RGB_image)  # apply gamma correction

    RGB_image = np.rot90(RGB_image, k=1, axes=(0, 1))

    return RGB_image


def calc_sto2(cube, gaussFkt, minWL=500, maxWL=995, WLsteps=5):
    # gaussian window for smoothing
    sigma = 2.66  # to be defined
    cube = gaussian_filter1d(cube, sigma, axis=2)
    #cube = convolve1d(cube, gaussFkt, axis=2, origin=0)

    first_derivative = np.gradient(cube, axis=2)
    second_derivative = np.gradient(first_derivative, axis=2)

    # calculate min of 2nd derivative between 573 and 587 nm
    first_range_start = int(round((573 - minWL) / WLsteps))
    first_range_end = int(round((587 - minWL) / WLsteps))
    min_value = second_derivative[:, :, first_range_start:first_range_end].min(axis=2)

    # calculate diff of mean and max of 2nd derivative between 740 and 780 nm
    second_range_start = int(round((740 - minWL) / WLsteps))
    second_range_end = int(round((780 - minWL) / WLsteps)) + 1  # !!! should be changed for diff WL steps
    max_value = second_derivative[:, :, second_range_start:second_range_end].max(axis=2)
    mean_value = second_derivative[:, :, second_range_start:second_range_end].mean(axis=2)

    # calculate parameter from range subresults
    sub1 = (min_value / 0.2)
    sub2 = ((mean_value - max_value) / (-0.03)) + sub1
    sub3 = ((sub1 / sub2) * 0.6) + 0.4
    sto2_image = np.exp(sub3) - 1.48

    # spatial smoothing with median kernel
    cv2.medianBlur(sto2_image, 5, dst=sto2_image)

    # clip values outside range
    np.clip(sto2_image, 0.000001, 1, out=sto2_image)

    sto2_image = np.rot90(sto2_image, k=1, axes=(0, 1))

    return sto2_image


def calc_nir(cube, minWL=500, maxWL=995, WLsteps=5):
    a = -0.46
    b = 0.45

    # calculate mean between 825 and 925 nm
    first_range_start = int((825 - minWL) / WLsteps)
    first_range_end = int((925 - minWL) / WLsteps)
    mean_value1 = cube[:, :, first_range_start:first_range_end].mean(axis=2)

    # calculate mean between 655 and 735 nm
    second_range_start = int((655 - minWL) / WLsteps)
    second_range_end = int((735 - minWL) / WLsteps)
    mean_value2 = cube[:, :, second_range_start:second_range_end].mean(axis=2)

    # calculate parameter from range subresults
    sub1 = (-np.log(mean_value1 / mean_value2) - a) / (b-a)
    nir_image = (np.log(sub1 + 2.51) / np.log(1.3)) - 3.8   # y = log1.3 (x + 2.51) - 3.8

    # clip values outside range
    np.clip(nir_image, 0.000001, 1, out=nir_image)
    nir_image = np.rot90(nir_image, k=1, axes=(0, 1))

    return nir_image


def calc_twi(cube, minWL=500, maxWL=995, WLsteps=5):
    a = 0.1
    b = -0.5
    # calculate mean between 875 and 895 nm
    first_range_start = int((875 - minWL) / WLsteps)
    first_range_end = int((895 - minWL) / WLsteps)
    mean_value1 = cube[:, :, first_range_start:first_range_end].mean(axis=2)

    # calculate mean between 950 and 975 nm
    second_range_start = int((950 - minWL) / WLsteps)
    second_range_end = int((975 - minWL) / WLsteps)
    mean_value2 = cube[:, :, second_range_start:second_range_end].mean(axis=2)

    # calculate parameter from range subresults
    twi_image = (-np.log(mean_value1 / mean_value2) - a) / (b-a)

    # clip values outside range
    np.clip(twi_image, 0.000001, 1, out=twi_image)
    twi_image = np.rot90(twi_image, k=1, axes=(0, 1))

    return twi_image


def calc_ohi(cube, minWL=500, maxWL=995, WLsteps=5):
    a = 0.4
    b = 1.55

    # calculate mean between 530 and 590 nm
    first_range_start = int((530 - minWL) / WLsteps)
    first_range_end = int((590 - minWL) / WLsteps)
    mean_value1 = cube[:, :, first_range_start:first_range_end].mean(axis=2)

    # calculate mean between 785 and 825 nm
    second_range_start = int((785 - minWL) / WLsteps)
    second_range_end = int((825 - minWL) / WLsteps)
    mean_value2 = cube[:, :, second_range_start:second_range_end].mean(axis=2)

    # calculate parameter from range subresults
    ohi_image = (-np.log(mean_value1 / mean_value2) - a) / (b-a) / 2

    # clip values outside range
    np.clip(ohi_image, 0.000001, 1, out=ohi_image)
    ohi_image = np.rot90(ohi_image, k=1, axes=(0, 1))

    return ohi_image


def param2rgb(param, color_map):
    param_255 = np.clip((param*255), 0, 255).astype(np.uint8)
    param_rgb = np.zeros((param_255.shape[0], param_255.shape[1], 3), dtype=np.uint8)

    # should work in one line with multidim LUT
    param_rgb[:, :, 0] = cv2.LUT(param_255, color_map[:, 0])    # red
    param_rgb[:, :, 1] = cv2.LUT(param_255, color_map[:, 1])    # green
    param_rgb[:, :, 2] = cv2.LUT(param_255, color_map[:, 2])    # blue

    return param_rgb
