import numpy as np
import pytest
import os
import tensorflow as tf

from data_utils.dataset.tfrecord.tfr_utils import filter_name_idx_and_labels, get_numpy_X
from ..conftest import (
    D1_X_0, D3_X_0
)
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)


def test_filter_name_idx_and_labels():
    X = tf.Variable([[i] * 10 for i in range(10)], dtype=tf.float32)
    y = tf.Variable([i % 4 for i in range(10)], dtype=tf.int64)
    sw = tf.Variable([weight for weight in range(10)], dtype=tf.float32)
    pat_idx = tf.Variable([i % 3 for i in range(10)], dtype=tf.int64)
    result_X = tf.Variable([[i] * 10 for i in [0, 1, 4, 6, 9]], dtype=tf.float32)
    result_y = tf.Variable([0, 1, 0, 2, 1], dtype=tf.int64)
    result_sw = tf.Variable([i for i in [0, 1, 4, 6, 9]], dtype=tf.float32)
    use_labels = tf.Variable([0, 1, 2], dtype=tf.int64)
    use_idx = tf.Variable([0, 1], dtype=tf.int64)
    datas = filter_name_idx_and_labels(X=X, y=y, sw=sw, pat_idx=pat_idx, use_pat_idx=use_idx, use_labels=use_labels)
    for data, result in zip(datas, (result_X, result_y, result_sw)):
        assert tf.math.reduce_all(data == result)


GET_NUMPY_X_DATA = [("1d", D1_X_0), ("3d", D3_X_0)]


@pytest.mark.parametrize("folder,result", GET_NUMPY_X_DATA)
def test_get_numpy_X(tfr_data_dir: str, folder: str, result: np.ndarray):
    tfr_path = os.path.join(tfr_data_dir, folder, "shuffled", "shuffle0" + TFR_FILE_EXTENSION)
    X = get_numpy_X(tfr_path=tfr_path)
    assert (X == result).all()
