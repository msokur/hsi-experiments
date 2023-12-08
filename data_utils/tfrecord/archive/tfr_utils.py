from typing import Dict, Union

import tensorflow as tf
import numpy as np

from configuration.parameter import (
    FEATURE_X, FEATURE_Y, FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2, FEATURE_WEIGHTS,
)


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_feature_keys() -> Dict[str, tf.io.FixedLenFeature]:
    """Keys to read raw TFRecord data."""
    keys_to_features = {
        FEATURE_X: tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value="NaN"),
        FEATURE_Y: tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value="NaN"),
        FEATURE_WEIGHTS: tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value="NaN"),
        FEATURE_SAMPLES: tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        FEATURE_SPEC: tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        FEATURE_X_AXIS_1: tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        FEATURE_X_AXIS_2: tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }

    return keys_to_features


def get_features(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> Dict[str, tf.train.Feature]:
    """Get the features to write data to a TFRecord dataset.
    Add also the number of samples and the size of X.

    Feature names:

    - X -> X
    - y -> y
    - sample_weights -> sample_weights
    - sample count -> samples
    - X last axis -> spectrum
    - if X patch sized: X axis 1 -> X_patch_0, X axis 2 -> X_patch_1

    :param X: Dataset samples to convert to feature
    :param y: Dataset labels to convert to feature
    :param sample_weights: Sample weights to convert to feature

    :return: Dictionary with TF features
    """
    X_shape = X.shape
    if X_shape.__len__() == 1 or X_shape.__len__() == 3 or X_shape.__len__() > 4:
        _error(var="X", org_shape=X_shape, shape="2 or 4 (patch size)")

    if y.shape.__len__() > 1:
        _error(var="y", org_shape=y.shape, shape="1")

    features = {
        FEATURE_X: _bytes_feature(value=X.astype(dtype=np.float32).tobytes()),
        FEATURE_Y: _bytes_feature(value=y.astype(dtype=np.int64).tobytes()),
        FEATURE_SAMPLES: _int64_feature(value=X_shape[0]),
        FEATURE_SPEC: _int64_feature(value=X_shape[-1])
    }

    if len(X_shape) > 2:
        features[FEATURE_X_AXIS_1] = _int64_feature(value=X_shape[1])
        features[FEATURE_X_AXIS_2] = _int64_feature(value=X_shape[2])

    if sample_weights is not None:
        if sample_weights.shape.__len__() > 1:
            _error(var="sample_weights", org_shape=sample_weights.shape, shape="1")
        features[FEATURE_WEIGHTS] = _bytes_feature(value=sample_weights.astype(dtype=np.float32).tobytes())

    return features


def tfr_parser(record, X_d3: bool, with_sw: bool):
    """Parse function for the raw TFRecord data.

    :param record: Raw TFRecord data
    :param X_d3: If True the data are patch sized
    :param with_sw: If True return withe sample weights

    :return: The parsed dataset
    """
    parsed = tf.io.parse_single_example(record, _get_feature_keys())
    # --- read and reshape X ---
    decode_X = tf.io.decode_raw(input_bytes=parsed[FEATURE_X], out_type=tf.float32)
    # --- get shape for X ---
    X_shape = [parsed[FEATURE_SAMPLES]]
    if X_d3:
        X_shape += [parsed[FEATURE_X_AXIS_1], parsed[FEATURE_X_AXIS_2]]
    X_shape += [parsed[FEATURE_SPEC]]
    X = tf.reshape(tensor=decode_X, shape=X_shape, name=FEATURE_X)

    # --- read and reshape y ---
    decode_y = tf.io.decode_raw(input_bytes=parsed[FEATURE_Y], out_type=tf.int64, name=FEATURE_Y)
    y = tf.reshape(tensor=decode_y, shape=[parsed[FEATURE_SAMPLES]], name=FEATURE_Y)

    # --- read sample weights ---
    if with_sw:
        sw = tf.io.decode_raw(input_bytes=parsed[FEATURE_WEIGHTS], out_type=tf.int64, name=FEATURE_WEIGHTS)
        return X, y, sw

    return X, y


def get_class_weights(dataset: tf.data.Dataset, labels: np.ndarray) -> Dict[Union[str, int], float]:
    """Calculated the class weights from a TFRecord dataset

    :param dataset: The TRRecord dataset
    :param labels: The labels to use for calculation

    :return: A dictionary with the label as key an the class weight as value
    """
    sums = tf.zeros(shape=labels.shape[0], dtype=tf.float32)

    indices = tf.reshape(tensor=tf.range(labels.shape[0]), shape=[labels.shape[0], 1])
    for data in dataset:
        addition = tf.map_fn(fn=lambda l: tf.reduce_sum(tf.cast(tf.equal(x=data[1], y=l), tf.float32), axis=0),
                             elems=labels, dtype=tf.float32)
        sums = tf.tensor_scatter_nd_add(tensor=sums, indices=indices, updates=addition)

    total = tf.reduce_sum(sums)
    weights = tf.math.divide_no_nan(x=1.0, y=sums)
    weights = weights * total / len(labels)

    return {label: weights.numpy()[idx] for idx, label in enumerate(labels)}


def _error(var: str, org_shape: tuple, shape: str):
    raise ValueError(f"'{var}' has wrong shape, the shape is {org_shape}. "
                     f"Only a shape with the length {shape} are allowed. Check your Data")
