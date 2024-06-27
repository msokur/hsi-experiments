import numpy as np

from ..conftest import (
    DATA_y_0,
    DATA_y_1
)

# --- Test data

SPEC = 15
SAMPLES = 100
PATCH = [3, 3]

D1_X_0 = np.array([[i] * SPEC for i in range(SAMPLES)], dtype=np.float32)
D1_X_1 = D1_X_0.copy() + SAMPLES

D3_X_0 = np.array([[i] * 3 * 3 * SPEC for i in range(SAMPLES)], dtype=np.float32).reshape([SAMPLES, PATCH[0],
                                                                                           PATCH[1], SPEC])
D3_X_1 = D3_X_0.copy() + SAMPLES

DATA_y_0_expand = np.expand_dims(a=DATA_y_0, axis=-1)
DATA_y_1_expand = np.expand_dims(a=DATA_y_1, axis=-1)

_y_list = []
for i, size in enumerate([20, 40, 10, 30]):
    _y_list += [i] * size
Y = np.array(_y_list, dtype=np.float32).reshape(-1)
# Y = np.expand_dims(a=np.array(_y_list, dtype=np.float32), axis=-1)

_sw_list = []
for weights, size in [(20.1, 20), (100.23, 40), (2.0, 10), (50.09, 30)]:
    _sw_list += [weights] * size
WEIGHTS = np.array(_sw_list, dtype=np.float32)

DATA_CUBE_IDX = np.array([[i, i] for i in range(SAMPLES)], dtype=np.int64)

_names = ["test_0", "test_1", "test_2", "test_3", "test_4"]
_names_list = []
_names_idx_list = []
for i in range(SAMPLES // 4):
    _names_list += [_names[i % _names.__len__()]] * 4
    _names_idx_list += [i % _names.__len__()] * 4
NAMES = np.array(_names_list)
NAMES_IDX = np.array(_names_idx_list, dtype=np.int64)

LABELS = [0, 1, 2]
USE_NAMES = ["test_0", "test_1", "test_2", "test_3"]

RES_MASK = np.isin(Y, LABELS).reshape(-1) * np.isin(NAMES_IDX, [0, 1, 2, 3])
BATCH_X_DATA_1D = np.concatenate((D1_X_0[RES_MASK], D1_X_1[RES_MASK]))
BATCH_X_DATA_3D = np.concatenate((D3_X_0[RES_MASK], D3_X_1[RES_MASK]))
BATCH_Y_DATA = np.concatenate((Y[RES_MASK], Y[RES_MASK]))
BATCH_SW_DATA = np.concatenate((WEIGHTS[RES_MASK], WEIGHTS[RES_MASK]))
