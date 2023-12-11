import pytest
import os
import numpy as np


@pytest.fixture
def tfr_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "tfr_file")


@pytest.fixture
def tfr_1d_data_dir(tfr_data_dir) -> str:
    return os.path.join(tfr_data_dir, "1d")


@pytest.fixture
def tfr_3d_data_dir(tfr_data_dir) -> str:
    return os.path.join(tfr_data_dir, "3d")


# --- Test data

SPEC = 15
SAMPLES = 100
PATCH = [3, 3]

DATA_1D_X_0 = np.array([[i] * SPEC for i in range(SAMPLES)], dtype=np.float32)

DATA_3D_X_0 = np.array([[i] * 3 * 3 * SPEC for i in range(SAMPLES)], dtype=np.float32).reshape([SAMPLES, PATCH[0],
                                                                                                PATCH[1], SPEC])

_y_list = []
for i, size in enumerate([20, 40, 10, 30]):
    _y_list += [i] * size
DATA_y_0 = np.array(_y_list, dtype=np.int64)

_sw_list = []
for weights, size in [(20.1, 20), (100.23, 40), (2.0, 10), (50.09, 30)]:
    _sw_list += [weights] * size
DATA_WEIGHTS_0 = np.array(_sw_list, dtype=np.float32)

DATA_i = np.array([[i, i] for i in range(SAMPLES)], dtype=np.int64)

_names = ["test_0", "test_1", "test_2", "test_3", "test_4"]
_names_list = []
_names_idx_list = []
for i in range(SAMPLES // 4):
    _names_list += [_names[i % _names.__len__()]] * 4
    _names_idx_list += [i % _names_list.__len__()] * 4
NAMES = np.array(_names_list)
NAMES_IDX = np.array(_names_idx_list, dtype=np.int64)
