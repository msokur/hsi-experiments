import tensorflow as tf

from configuration.parameter import (
    FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_X, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2, FEATURE_Y, FEATURE_WEIGHTS,
    FEATURE_PAT_IDX)


def tfr_1d_train_parser(record):
    parsed = tf.io.parse_single_example(record, _base_feature_keys())

    # --- read and reshape X ---
    X_shape = [parsed[FEATURE_SAMPLES], parsed[FEATURE_SPEC]]
    X = _parse_and_reshape_X(input_bytes=parsed[FEATURE_X], shape=X_shape)
    # --- read y, sample weight and patient index
    y, sw, pat_idx = _parse_y_sw_and_pat_idx(parsed=parsed)

    return X, y, sw, pat_idx


def tfr_3d_train_parser(record):
    parsed = tf.io.parse_single_example(record, _base_3d_feature_keys())

    # --- read and reshape X ---
    X_shape = [parsed[FEATURE_SAMPLES], parsed[FEATURE_X_AXIS_1], parsed[FEATURE_X_AXIS_2], parsed[FEATURE_SPEC]]
    X = _parse_and_reshape_X(input_bytes=parsed[FEATURE_X], shape=X_shape)
    # --- read y, sample weight and patient index
    y, sw, pat_idx = _parse_y_sw_and_pat_idx(parsed=parsed)

    return X, y, sw, pat_idx


def _base_feature_keys():
    return {
        FEATURE_X: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_Y: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_WEIGHTS: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_PAT_IDX: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_SAMPLES: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        FEATURE_SPEC: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }


def _base_3d_feature_keys():
    feature_keys = _base_feature_keys()
    feature_keys[FEATURE_X_AXIS_1] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    feature_keys[FEATURE_X_AXIS_2] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)

    return feature_keys


def _parse_and_reshape_X(input_bytes, shape):
    return tf.reshape(tensor=_float32_decode_raw(input_bytes=input_bytes), shape=shape, name=FEATURE_X)


def _parse_y_sw_and_pat_idx(parsed):
    y = _int64_decode_raw(input_bytes=parsed[FEATURE_Y])
    sw = _float32_decode_raw(input_bytes=parsed[FEATURE_WEIGHTS])
    pat_idx = _int64_decode_raw(input_bytes=parsed[FEATURE_PAT_IDX])

    return y, sw, pat_idx


def _float32_decode_raw(input_bytes):
    """Returns an object with float32 numbers"""
    return tf.io.decode_raw(input_bytes=input_bytes, out_type=tf.float32)


def _int64_decode_raw(input_bytes):
    """Returns an object with int64 numbers"""
    return tf.io.decode_raw(input_bytes=input_bytes, out_type=tf.int64)
