import pytest
import os
import tensorflow as tf
import numpy as np

from data_utils.tfrecord.tfr_utils import get_features, tfr_parser, get_class_weights
from configuration.parameter import (
    FEATURE_X, FEATURE_Y, FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2, FEATURE_WEIGHTS,
)
from testspace.data_utils.conftest import (
    DATA_1D_X_0, DATA_1D_X_1, DATA_3D_X_0, DATA_3D_X_1, DATA_y_0, DATA_y_1, DATA_WEIGHTS_0, DATA_WEIGHTS_1,
)


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


GET_FEATURES_LENGTH_DATA = [({"X": DATA_1D_X_0, "y": DATA_y_0}, 4),
                            ({"X": DATA_1D_X_0, "y": DATA_y_0, "sample_weights": DATA_WEIGHTS_0}, 5),
                            ({"X": DATA_3D_X_0, "y": DATA_y_0}, 6),
                            ({"X": DATA_3D_X_0, "y": DATA_y_0, "sample_weights": DATA_WEIGHTS_0}, 7)]


@pytest.mark.parametrize("args,result", GET_FEATURES_LENGTH_DATA)
def test_get_features_length(args: dict, result: int):
    features = get_features(**args)
    assert features.__len__() == result


X1D_BYTE_LIST = _bytes_feature(value=DATA_1D_X_0.astype(dtype=np.float32).tobytes())
X3D_BYTE_LIST = _bytes_feature(value=DATA_3D_X_0.astype(dtype=np.float32).tobytes())
Y_BYTE_LIST = _bytes_feature(value=DATA_y_0.astype(dtype=np.int64).tobytes())
WEIGHTS_BYTE_LIST = _bytes_feature(value=DATA_WEIGHTS_0.astype(dtype=np.int64).tobytes())
SAMPLES_INT_LIST = _int64_feature(value=12)
SPEC_INT_LIST = _int64_feature(value=10)
PATCH_INT_LIST = _int64_feature(value=3)

GET_FEATURES_VALUE_DATA = [({"X": DATA_1D_X_0, "y": DATA_y_0},
                            {FEATURE_X: X1D_BYTE_LIST, FEATURE_Y: Y_BYTE_LIST, FEATURE_SAMPLES: SAMPLES_INT_LIST,
                             FEATURE_SPEC: SPEC_INT_LIST}),
                           ({"X": DATA_1D_X_0, "y": DATA_y_0, "sample_weights": DATA_WEIGHTS_0},
                            {FEATURE_X: X1D_BYTE_LIST, FEATURE_Y: Y_BYTE_LIST, FEATURE_SAMPLES: SAMPLES_INT_LIST,
                             FEATURE_SPEC: SPEC_INT_LIST, FEATURE_WEIGHTS: WEIGHTS_BYTE_LIST}),
                           ({"X": DATA_3D_X_0, "y": DATA_y_0},
                            {FEATURE_X: X3D_BYTE_LIST, FEATURE_Y: Y_BYTE_LIST, FEATURE_SAMPLES: SAMPLES_INT_LIST,
                             FEATURE_SPEC: SPEC_INT_LIST, FEATURE_X_AXIS_1: PATCH_INT_LIST,
                             FEATURE_X_AXIS_2: PATCH_INT_LIST}),
                           ({"X": DATA_3D_X_0, "y": DATA_y_0, "sample_weights": DATA_WEIGHTS_0},
                            {FEATURE_X: X3D_BYTE_LIST, FEATURE_Y: Y_BYTE_LIST, FEATURE_SAMPLES: SAMPLES_INT_LIST,
                             FEATURE_SPEC: SPEC_INT_LIST, FEATURE_X_AXIS_1: PATCH_INT_LIST,
                             FEATURE_X_AXIS_2: PATCH_INT_LIST, FEATURE_WEIGHTS: WEIGHTS_BYTE_LIST})]


@pytest.mark.parametrize("args,result", GET_FEATURES_VALUE_DATA)
def test_get_features_length(args: dict, result: dict):
    features = get_features(**args)
    for (k_feat, v_feat), (k_result, v_result) in zip(features.items(), result.items()):
        assert k_feat == k_result and v_feat == v_result


GET_FEATURES_ERROR_DATA = [({"X": np.array([0, 1, 2, 3]), "y": DATA_y_0}),
                           ({"X": np.arange(27).reshape((3, 3, 3)), "y": DATA_y_0}),
                           ({"X": np.arange(243).reshape((3, 3, 3, 3, 3)), "y": DATA_y_0}),
                           ({"X": DATA_1D_X_0, "y": np.array([[0, 1], [2, 3]])}),
                           ({"X": DATA_1D_X_0, "y": DATA_y_0, "sample_weights": np.array([[0, 1, 2], [3, 4, 5]])})]


@pytest.mark.parametrize("args", GET_FEATURES_ERROR_DATA)
def test_get_features_error(args: dict):
    with pytest.raises(ValueError):
        get_features(**args)


TFR_PARSER_DATA = [("1d", False, [(DATA_1D_X_0, DATA_y_0), (DATA_1D_X_1, DATA_y_1)]),
                   ("1d", True, [(DATA_1D_X_0, DATA_y_0, DATA_WEIGHTS_0), (DATA_1D_X_1, DATA_y_1, DATA_WEIGHTS_1)]),
                   ("3d", False, [(DATA_3D_X_0, DATA_y_0), (DATA_3D_X_1, DATA_y_1)]),
                   ("3d", True, [(DATA_3D_X_0, DATA_y_0, DATA_WEIGHTS_0), (DATA_3D_X_1, DATA_y_1, DATA_WEIGHTS_1)])]


@pytest.mark.parametrize("shape,with_sw,result", TFR_PARSER_DATA)
def test_tfr_parser(tfr_data_dir: str, tfr_file_name: str, shape: str, with_sw: bool, result):
    dataset = tf.data.TFRecordDataset(filenames=os.path.join(tfr_data_dir, shape, tfr_file_name))
    dataset = dataset.map(map_func=lambda record: tfr_parser(record=record, X_d3=False if shape == "1d" else True,
                                                             with_sw=with_sw))
    for d_set, result_set in zip(dataset, result):
        for d_val, result_val in zip(d_set, result_set):
            assert (d_val.numpy() == result_val).all()


def test_tfr_parser_error(tfr_1d_data_dir: str, tfr_file_name: str):
    dataset = tf.data.TFRecordDataset(filenames=os.path.join(tfr_1d_data_dir, tfr_file_name))
    dataset = dataset.map(map_func=lambda record: tfr_parser(record=record, X_d3=True, with_sw=False))
    with pytest.raises(tf.errors.InvalidArgumentError):
        for _, _ in dataset:
            pass


GET_CLASS_WEIGHTS_DATA = [(DATA_y_0, np.array([0, 1, 2, 3]), {0: 1., 1: 1., 2: 1., 3: 1.}),
                          (DATA_y_1, np.array([0, 2, 3]), {0: 0.33, 2: 0., 3: 0.})]


@pytest.mark.parametrize("y,labels,result", GET_CLASS_WEIGHTS_DATA)
def test_get_class_weights(y: np.ndarray, labels: np.ndarray, result: dict):
    dataset = tf.data.Dataset.from_tensor_slices((DATA_3D_X_0, y)).batch(3)
    weights = get_class_weights(dataset=dataset, labels=labels)
    for (k_ds, v_ds), (k_r, v_r) in zip(weights.items(), result.items()):
        assert k_ds == k_r and v_ds == pytest.approx(v_ds, rel=1e-2)
