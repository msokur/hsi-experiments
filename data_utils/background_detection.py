import numpy as np
# import cv2
from scipy.ndimage import gaussian_filter1d

# import sklearn
# from sklearn.preprocessing import StandardScaler
# import joblib   # (version 0.14.1)
backend = "threading"


# sklearn.utils.parallel_backend(backend, n_jobs=-1)


def detect_background(cube, minWL=500, maxWL=995, WLsteps=5, cube_index=None, bg_mask=None, scanning='horizontal'):
    a = -0.1
    b = 1.6
    first_range_start = int((510 - minWL) / WLsteps)
    first_range_end = int((570 - minWL) / WLsteps)
    second_range_start = int((650 - minWL) / WLsteps)
    second_range_end = int((710 - minWL) / WLsteps)

    if cube_index is None:
        # calculate mean between 510 and 570 nm
        mean_value1 = None
        # calculate mean between 650 and 710 nm
        mean_value2 = None
        # calculate mean between 500 and 1000 nm
        mean_value3 = None

        if len(cube.shape) == 3:
            mean_value1 = cube[:, :, first_range_start:first_range_end].mean(axis=2)
            mean_value2 = cube[:, :, second_range_start:second_range_end].mean(axis=2)
            mean_value3 = cube[:, :, :].mean(axis=2)
        elif len(cube.shape) == 2:
            mean_value1 = cube[:, first_range_start:first_range_end].mean(axis=1)
            mean_value2 = cube[:, second_range_start:second_range_end].mean(axis=1)
            mean_value3 = cube[:, :].mean(axis=1)

        # calculate parameter from range subresults
        sub1 = (-np.log(mean_value1 / mean_value2) - a) / b

        mean_value1[mean_value1 > 0.25] = 0
        sub1[sub1 < 0] = 0
        mean_value3[mean_value3 < 0.1] = 0

        bg_mask = mean_value1 * sub1 * mean_value3
        bg_mask[bg_mask != 0] = 1
        # bg_mask = np.rot90(bg_mask, k=1, axes=(0, 1))

    else:
        # calculate mean between 510 and 570 nm
        mean_value1 = cube[:, first_range_start:first_range_end].mean(axis=1)
        # calculate mean between 650 and 710 nm
        mean_value2 = cube[:, second_range_start:second_range_end].mean(axis=1)
        # calculate parameter from range subresults
        sub1 = (-np.log(mean_value1 / mean_value2) - a) / b
        # calculate mean between 500 and 1000 nm
        mean_value3 = cube[:, :].mean(axis=1)

        mean_value1[mean_value1 > 0.7] = 0
        sub1[sub1 < 0] = 0
        mean_value3[mean_value3 < 0.1] = 0

        bg_mask_line = mean_value1 * sub1 * mean_value3
        bg_mask_line[bg_mask_line != 0] = 1

        if scanning == 'horizontal':  # scanning left2right
            bg_mask[:, cube_index] = bg_mask_line[::-1]
        else:  # scanning bottom2top
            bg_mask[cube_index, :] = bg_mask_line[::-1]

    return bg_mask.astype(bool)
