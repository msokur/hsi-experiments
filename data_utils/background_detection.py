# Original background detection
# The only thing that was changed
# Thresholds in lines 42 and 44: 0.25 for light reflections and 0.1 for blood,

import numpy as np


def detect_background(cube, blood_threshold=0.1, lights_reflections_threshold=0.25):
    a = -0.1
    b = 1.6
    minWL = 500
    maxWL = 995
    WLsteps = 5

    first_range_start = int((510 - minWL) / WLsteps)
    first_range_end = int((570 - minWL) / WLsteps)
    second_range_start = int((650 - minWL) / WLsteps)
    second_range_end = int((710 - minWL) / WLsteps)

    # calculate mean between 510 and 570 nm
    mean_value1 = cube[:, :, first_range_start:first_range_end].mean(axis=2)
    # calculate mean between 650 and 710 nm
    mean_value2 = cube[:, :, second_range_start:second_range_end].mean(axis=2)
    # calculate mean between 500 and 1000 nm
    mean_value3 = cube[:, :, :].mean(axis=2)

    # calculate parameter from range subresults
    sub1 = (-np.log(mean_value1 / mean_value2) - a) / b

    mean_value1[mean_value1 > lights_reflections_threshold] = 0
    sub1[sub1 < 0] = 0
    mean_value3[mean_value3 < blood_threshold] = 0

    bg_mask = mean_value1 * sub1 * mean_value3
    bg_mask[bg_mask != 0] = 1

    return bg_mask.astype(bool)
