from typing import Dict

import numpy as np
import tensorflow as tf

from data_utils.dataset.tfrecord.tfr_parser import tfr_X_parser
from data_utils.dataset.meta_files import get_meta_files, get_section_from_meta
from configuration.parameter import (
    SAMPLES_PER_NAME, FEATURE_PAT_IDX, TFR_TYP
)


def parse_names_to_int(tfr_files: list) -> Dict[str, int]:
    """Read the integer values in the metafile for every name.

    :param tfr_files: Paths to datasets

    :return: A dictionary with the actual str name as key and the integer as value

    :raises ValueError: If there is more than one integer for a name
    """
    meta_files = get_meta_files(paths=tfr_files, typ=TFR_TYP)

    names_int = {}
    for meta_file in meta_files:
        samples_per_names = get_section_from_meta(file_path=meta_file, section=SAMPLES_PER_NAME)
        for name in samples_per_names.keys():
            # get indexes from all meta files
            if name not in names_int:
                names_int[name] = [samples_per_names[name][FEATURE_PAT_IDX]]
            else:
                names_int[name] += [samples_per_names[name][FEATURE_PAT_IDX]]

    for name, indexes in names_int.items():
        unique_idx = np.unique(indexes)
        if unique_idx.shape[0] == 1:
            names_int[name] = int(unique_idx[0])
        else:
            raise ValueError(f"Too many patient indexes in meta files for the name '{name}'!")

    return names_int


def filter_name_idx_and_labels(X, y, sw, pat_idx, use_pat_idx: tf.Variable, use_labels: tf.Variable):
    """Filter function for a tfrecord dataset

     Filters the variables X, y and sw by patient indexes and labels.

    :param X: List with trainings data
    :param y: List with labels
    :param sw: List with sample weights
    :param pat_idx: List with patient indexes
    :param use_pat_idx: patient index to filter
    :param use_labels: labels to filter

    :return: Filtered X, y and sw
    """
    idx_mask = _select(data=pat_idx, allowed=use_pat_idx)
    label_mask = _select(data=y, allowed=use_labels)
    mask = tf.logical_and(x=idx_mask, y=label_mask)

    return (tf.boolean_mask(tensor=X, mask=mask), tf.boolean_mask(tensor=y, mask=mask),
            tf.boolean_mask(tensor=sw, mask=mask))


def get_numpy_X(tfr_path: str, shape: tuple) -> np.ndarray:
    """Get X as numpy array from a TFRecord file.

    :param tfr_path: Path from the file to read
    :param shape: The shape from X

    :return: Numpy array with data from X
    """
    shape_ = tf.Variable(shape, dtype=tf.int64)
    data = tf.data.TFRecordDataset(filenames=tfr_path).map(map_func=lambda record: tfr_X_parser(record=record,
                                                                                                shape=shape_))

    return data.as_numpy_iterator().get_next()


def _select(data: tf.Variable, allowed: tf.Variable):
    # get a boolean list for every element in allowed
    use = tf.map_fn(fn=lambda com: tf.equal(x=data, y=com), elems=allowed, fn_output_signature=tf.bool)
    # combine the lists with a logical or
    return tf.cast(tf.math.reduce_sum(input_tensor=tf.cast(use, dtype=tf.int32), axis=0), dtype=tf.bool)
