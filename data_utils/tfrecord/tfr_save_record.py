from typing import Dict

import tensorflow as tf
import numpy as np
import os

from configuration.parameter import (
    TFR_FILE_EXTENSION, FEATURE_X, FEATURE_Y, FEATURE_PAT_NAME, PAT_NAME_SEPERATOR, PAT_NAME_ENCODING, FEATURE_PAT_IDX,
    FEATURE_IDX_CUBE, FEATURE_WEIGHTS, FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2,
)


def save_tfr_file(save_path: str, file_name: str, X: np.ndarray, y: np.ndarray, pat_names: np.ndarray,
                  pat_idx: np.ndarray, idx_in_cube: np.ndarray, sw: np.ndarray):
    """ Store a TFRecord file.

        :param save_path: Parent path for file.
        :param file_name: file name to store the data.
        :param X: Array with samples to convert to feature.
        :param y: Array with labels to convert to feature.
        :param pat_names: Array with patient names to convert to feature.
        :param pat_idx: Array with patient indexes to convert to feature.
        :param idx_in_cube: Array with original indexes in datacube to convert to feature.
        :param sw: Array with sample weights to convert to feature.
    """
    file_path = os.path.join(save_path, file_name)
    if not file_path.endswith(TFR_FILE_EXTENSION):
        file_path += TFR_FILE_EXTENSION

    writer = tf.io.TFRecordWriter(path=file_path)

    features = _get_features(X=X, y=y, pat_names=pat_names, pat_idx=pat_idx, idx_in_cube=idx_in_cube, sw=sw)
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
    writer.close()


def _get_features(X: np.ndarray, y: np.ndarray, pat_names: np.ndarray, pat_idx: np.ndarray, idx_in_cube: np.ndarray,
                  sw: np.ndarray) -> Dict[str, tf.train.Feature]:
    """Get the features to write data to a TFRecord dataset.
        Add also the number of samples and the size of X.

        Feature names:

        - X -> X
        - y -> y
        - patient names -> patient_names
        - patient path index -> patient_index
        - original index in cube -> indexes_in_datacube
        - sample_weights -> sample_weights
        - sample count -> samples
        - X last axis -> spectrum
        - if X patch sized: X axis 1 -> X_patch_0, X axis 2 -> X_patch_1

        :param X: Array with samples to convert to feature.
        :param y: Array with labels to convert to feature.
        :param pat_names: Array with patient names to convert to feature.
        :param pat_idx: Array with patient indexes to convert to feature.
        :param idx_in_cube: Array with original indexes in datacube to convert to feature.
        :param sw: Array with sample weights to convert to feature.

        :return: Dictionary with TF features.

        :raises ValueError: If the shape of an array is incorrect.
        """
    X_shape = X.shape
    if not (X_shape.__len__() == 2 or X_shape.__len__() == 4):
        _error(var="X", org_shape=X_shape, shape="2 or 4 (patch size)")

    if y.shape.__len__() > 1:
        _error(var="y", org_shape=y.shape, shape="1")

    if sw.shape.__len__() > 1:
        _error(var="sample_weights", org_shape=sw.shape, shape="1")

    features = {
        FEATURE_X: _bytes_feature(value=X.astype(dtype=np.float32).tobytes()),
        FEATURE_Y: _bytes_feature(value=y.astype(dtype=np.int64).tobytes()),
        FEATURE_PAT_NAME: _bytes_feature(value=PAT_NAME_SEPERATOR.join([n for n in pat_names])
                                         .encode(encoding=PAT_NAME_ENCODING)),
        FEATURE_PAT_IDX: _bytes_feature(value=pat_idx.astype(dtype=np.int64).tobytes()),
        FEATURE_IDX_CUBE: _bytes_feature(value=idx_in_cube.astype(dtype=np.int64).tobytes()),
        FEATURE_WEIGHTS: _bytes_feature(value=sw.astype(dtype=np.float32).tobytes()),
        FEATURE_SAMPLES: _int64_feature(value=X_shape[0]),
        FEATURE_SPEC: _int64_feature(value=X_shape[-1])
    }

    if len(X_shape) > 2:
        features[FEATURE_X_AXIS_1] = _int64_feature(value=X_shape[1])
        features[FEATURE_X_AXIS_2] = _int64_feature(value=X_shape[2])

    return features


def _error(var: str, org_shape: tuple, shape: str):
    raise ValueError(f"'{var}' has wrong shape, the shape is {org_shape}. "
                     f"Only a shape with the length {shape} are allowed. Check your Data")


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
