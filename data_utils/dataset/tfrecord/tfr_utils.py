import numpy as np
import tensorflow as tf

from data_utils.dataset.tfrecord.tfr_parser import tfr_X_parser


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

    return data.as_numpy_iterator().__next__()


def _select(data: tf.Variable, allowed: tf.Variable):
    # get a boolean list for every element in allowed
    use = tf.map_fn(fn=lambda com: tf.equal(x=data, y=com), elems=allowed, fn_output_signature=tf.bool)
    # combine the lists with a logical or
    return tf.cast(tf.math.reduce_sum(input_tensor=tf.cast(use, dtype=tf.int32), axis=0), dtype=tf.bool)
