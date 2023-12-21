import pytest
import os
import tensorflow as tf

from glob import glob

from data_utils.tfrecord.tfr_utils import get_cw_from_meta, parse_names_to_int, filter_name_idx_and_labels
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)


CW_TEST_DATA = [("1d", [0, 1, 2, 3], ["test_0", "test_1", "test_2", "test_3", "test_4"],
                 {"0": 1.25, "1": 0.625, "2": 2.5, "3": 0.833}),
                ("1d", [0, 1, 3], ["test_0", "test_1", "test_2", "test_3"], {"0": 1.458, "1": 0.729, "3": 1.061}),
                ("3d", [1, 2, 3], ["test_0", "test_2", "test_3", "test_4"], {"1": 0.667, "2": 3.556, "3": 0.821}),
                ("1d", [0, 1, 2, 3, 4], ["test_0", "test_1", "test_2", "test_3", "test_4"],
                 {"0": 1., "1": 0.5, "2": 2., "3": 0.667, "4": 0.0})]


@pytest.mark.parametrize("shape,labels,names,result", CW_TEST_DATA)
def test_cw_from_meta(tfr_data_dir: str, shape: str, labels: list, names: list, result: dict):
    stored_path = glob(os.path.join(tfr_data_dir, shape, "*" + TFR_FILE_EXTENSION))
    cw = get_cw_from_meta(tfr_files=stored_path, labels=labels, names=names)
    for (cw_l, cw_v), (res_l, res_v) in zip(cw.items(), result.items()):
        assert cw_l == res_l and pytest.approx(cw_v, rel=1e-3) == res_v


def test_cw_from_meta_error():
    with pytest.raises(FileNotFoundError, match="No such file or directory: 'test'"):
        get_cw_from_meta(tfr_files=["test"], labels=[], names=[])


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
