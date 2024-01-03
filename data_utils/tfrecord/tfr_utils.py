import json

from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

from data_utils.tfrecord.tfr_parser import tfr_X_parser
from configuration.parameter import (
    TFR_META_EXTENSION, TFR_FILE_EXTENSION, SAMPLES_PER_NAME, FEATURE_PAT_IDX, FEATURE_X
)


def get_cw_from_meta(tfr_files: list, labels: list, names: list) -> Dict[str, float]:
    """Calculate class weights from meta file.

    :param tfr_files: Paths to datasets
    :param labels: Used labels
    :param names: Used names

    :return: A dictionary with the label name as str and the class weight
    """
    meta_files = _get_meta_files(tfr_paths=tfr_files)

    sums = {f"{label}": 0.0 for label in labels}

    for samples_per_labels in _get_samples_per_label(paths=meta_files, names=names):
        for label, v in samples_per_labels.items():
            if label in sums.keys():
                sums[label] += v

    total = np.sum([samples for samples in sums.values()])
    cw = {}
    for label, samples in sums.items():
        if samples > 0.0:
            cw[label] = (1 / samples) * total / labels.__len__()
        else:
            cw[label] = 0.0

    return cw


def parse_names_to_int(tfr_files: list) -> Dict[str, int]:
    """Read the integer values in the metafile for every name.

    :param tfr_files: Paths to datasets

    :return: A dictionary with the actual str name as key and the integer as value

    :raises ValueError: If there is more than one integer for a name
    """
    meta_files = _get_meta_files(tfr_paths=tfr_files)

    names_int = {}
    for meta_file in meta_files:
        samples_per_names = _get_section_from_meta(file_path=meta_file, section=SAMPLES_PER_NAME)
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
    idx_mask = _select(data=pat_idx, allowed=use_pat_idx)
    label_mask = _select(data=y, allowed=use_labels)
    mask = tf.logical_and(x=idx_mask, y=label_mask)

    return (tf.boolean_mask(tensor=X, mask=mask), tf.boolean_mask(tensor=y, mask=mask),
            tf.boolean_mask(tensor=sw, mask=mask))


def get_shape_from_meta(tfr_files: list) -> Tuple[int]:
    """Get Dataset shape for X from meta files

    :param tfr_files: Paths to datasets

    :return: A tuple with the shape for X

    :raises ValueError: When different shape in the meta files
    """
    meta_files = _get_meta_files(tfr_paths=tfr_files)

    model_shape = None
    for p in meta_files:
        if model_shape is None:
            model_shape = _get_section_from_meta(file_path=p, section=f"{FEATURE_X}_shape")
        else:
            if model_shape != _get_section_from_meta(file_path=p, section=f"{FEATURE_X}_shape"):
                raise ValueError("Chck your dataset! Different data shapes in meta files!")

    return tuple(model_shape)


def get_numpy_X(tfr_path: str, shape: tuple) -> np.ndarray:
    """Get X as numpy array from a TFRecord file.

    :param tfr_path: Path from the file to read
    :param shape: The shape from X

    :return: Numpy array with data from X
    """
    shape_ = tf.Variable(shape, dtype=tf.int64)
    data = tf.data.TFRecordDataset(filenames=tfr_path).map(map_func=lambda record: tfr_X_parser(record=record,
                                                                                                shape=shape_))

    return data.as_numpy_iterator().next()


def _get_meta_files(tfr_paths: List[str]) -> List[str]:
    """Replace TFRecord file extension with TFRecord meta file extension.

    :param tfr_paths: TFRecord file paths

    :return: List with paths to TFRecord meta files
    """
    return [tfr_p.replace(TFR_FILE_EXTENSION, TFR_META_EXTENSION) for tfr_p in tfr_paths]


def _get_samples_per_label(paths: List[str], names: list):
    for p in paths:
        samples_per_names = _get_section_from_meta(file_path=p, section=SAMPLES_PER_NAME)
        # create a list with samples per label for the needed names
        samples_per_labels = [samples_per_names[name] for name in names if name in names]
        for samples_per_label in samples_per_labels:
            # remove patient index
            samples_per_label.pop(FEATURE_PAT_IDX)
            yield samples_per_label


def _get_section_from_meta(file_path: str, section: str) -> dict:
    # open meta file
    info = json.load(open(file=file_path, mode="r"))
    # get the section with the samples per name
    return info[section]


def _select(data, allowed):
    use = tf.map_fn(fn=lambda com: tf.equal(x=data, y=com), elems=allowed, fn_output_signature=tf.bool)
    return tf.cast(tf.math.reduce_sum(input_tensor=tf.cast(use, dtype=tf.int32), axis=0), dtype=tf.bool)
