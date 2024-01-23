import pytest
import os
import tensorflow as tf

from glob import glob

from data_utils.dataset.tfrecord.tfr_utils import parse_names_to_int, filter_name_idx_and_labels
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)


def test_parse_names_to_int(tfr_1d_data_dir: str):
    names_int = parse_names_to_int(tfr_files=sorted(glob(os.path.join(tfr_1d_data_dir, "*" + TFR_FILE_EXTENSION))))
    assert names_int == {"test_0": 0, "test_1": 1, "test_2": 2, "test_3": 3, "test_4": 4}


def test_parse_names_to_int_error(tfr_data_dir: str):
    with pytest.raises(ValueError, match="Too many patient indexes in meta files for the name 'test_4'!"):
        parse_names_to_int(tfr_files=[os.path.join(tfr_data_dir, "meta_files_error", f"shuffle{i}{TFR_FILE_EXTENSION}")
                                      for i in [0, 1]])


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
