import tensorflow as tf

from configuration.parameter import (
    FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_X, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2, FEATURE_Y, FEATURE_WEIGHTS,
    FEATURE_PAT_IDX)


def tfr_1d_train_parser(record):
    parsed = tf.io.parse_single_example(record, _base_feature_keys())

    # --- read and reshape X ---
    X_shape = [parsed[FEATURE_SAMPLES], parsed[FEATURE_SPEC]]
    X = _decode_and_reshape_X(parsed=parsed, shape=X_shape)
    # --- read y, sample weight and patient index
    y, sw, pat_idx = _parse_y_sw_and_pat_idx(parsed=parsed)

    return X, y, sw, pat_idx


def tfr_3d_train_parser(record):
    parsed = tf.io.parse_single_example(record, _base_3d_feature_keys())

    # --- read and reshape X ---
    X_shape = [parsed[FEATURE_SAMPLES], parsed[FEATURE_X_AXIS_1], parsed[FEATURE_X_AXIS_2], parsed[FEATURE_SPEC]]
    X = _decode_and_reshape_X(parsed=parsed, shape=X_shape)
    # --- read y, sample weight and patient index
    y, sw, pat_idx = _parse_y_sw_and_pat_idx(parsed=parsed)

    return X, y, sw, pat_idx


def tfr_X_parser(record):
    parsed = tf.io.parse_single_example(record, _base_3d_feature_keys())

    if tf.equal(parsed[FEATURE_X_AXIS_1], -1):
        X_shape = tf.concat(values=[[parsed[FEATURE_SAMPLES]], [parsed[FEATURE_SPEC]]], axis=0)
    else:
        X_shape = tf.concat(values=[[parsed[FEATURE_SAMPLES]], [parsed[FEATURE_X_AXIS_1]], [parsed[FEATURE_X_AXIS_2]],
                                    [parsed[FEATURE_SPEC]]], axis=0)

    return _decode_and_reshape_X(parsed=parsed, shape=X_shape)


def _base_feature_keys():
    """Returns the feature keys for a 1D dataset"""
    return {
        FEATURE_X: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_Y: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_WEIGHTS: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_PAT_IDX: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        FEATURE_SAMPLES: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        FEATURE_SPEC: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }


def _base_3d_feature_keys():
    """Returns the feature keys for a 3D dataset"""
    feature_keys = _base_feature_keys()
    feature_keys[FEATURE_X_AXIS_1] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1)
    feature_keys[FEATURE_X_AXIS_2] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1)

    return feature_keys


def _decode_and_reshape_X(parsed, shape):
    """Decode and reshape X and returns it.

    :param parsed: Parsed bytes with keys for all values and raw bytes
    :param shape: Shape for X

    :returns: TF object with float32 numbers
    """
    decoded_X = _float32_decode_raw(input_bytes=parsed[FEATURE_X], name=FEATURE_X)
    return tf.reshape(tensor=decoded_X, shape=shape, name=FEATURE_X)


def _parse_y_sw_and_pat_idx(parsed):
    """Return three objects for labels, sample weights and patient indexes

    :param parsed: Parsed bytes with keys for all values and raw bytes

    :returns: Three TF objects. y (labels, int64), sw (sample weights, float32) and pat_idx (patient indexes, int64)
    """
    y = _int64_decode_raw(input_bytes=parsed[FEATURE_Y], name=FEATURE_Y)
    sw = _float32_decode_raw(input_bytes=parsed[FEATURE_WEIGHTS], name=FEATURE_WEIGHTS)
    pat_idx = _int64_decode_raw(input_bytes=parsed[FEATURE_PAT_IDX], name=FEATURE_PAT_IDX)

    return y, sw, pat_idx


def _float32_decode_raw(input_bytes, name: str):
    """Returns an object with float32 numbers

    :param input_bytes: Bytes to decode as float32 number
    :param name: Name for the object

    :returns: A TF object with float32 numbers
    """
    return tf.io.decode_raw(input_bytes=input_bytes, out_type=tf.float32, name=name)


def _int64_decode_raw(input_bytes, name: str):
    """Returns an object with int64 numbers

    :param input_bytes: Bytes to decode as int64 number
    :param name: Name for the object

    :returns: A TF object with int64 numbers
    """
    return tf.io.decode_raw(input_bytes=input_bytes, out_type=tf.int64, name=name)
