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


def _get_feature_keys():
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
    X_shape = X.shape

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
        features[FEATURE_WEIGHTS] = _bytes_feature(value=sample_weights.astype(dtype=np.int64).tobytes())

    return features


def tfr_parser(record, X_d3: bool, with_sw: bool):
    parsed = tf.io.parse_single_example(record, _get_feature_keys())
    # --- read and reshape X ---
    decode_X = tf.io.decode_raw(input_bytes=parsed[FEATURE_X], out_type=tf.float32)
    # --- get shape for X ---
    X_shape = [parsed[FEATURE_SAMPLES]]
    if X_d3:
        X_shape += [parsed[FEATURE_X_AXIS_1], [FEATURE_X_AXIS_2]]
    X_shape += [parsed[FEATURE_X_AXIS_2]]
    X = tf.reshape(tensor=decode_X, shape=X_shape, name=FEATURE_X)

    # --- read and reshape y ---
    decode_y = tf.io.decode_raw(input_bytes=parsed[FEATURE_Y], out_type=tf.int64, name=FEATURE_Y)
    y = tf.reshape(tensor=decode_y, shape=[parsed[FEATURE_SAMPLES]], name=FEATURE_Y)

    # --- read sample weights
    if with_sw:
        sw = tf.io.decode_raw(input_bytes=parsed[FEATURE_WEIGHTS], out_type=tf.int64, name=FEATURE_WEIGHTS)
        return X, y, sw

    return X, y


def get_class_weights(dataset: tf.data.Dataset, labels: np.ndarray) -> Dict[Union[str, int], float]:
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
