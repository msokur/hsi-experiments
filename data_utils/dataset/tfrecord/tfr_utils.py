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


def filter_labels_by_split_factor(X, y, sw, use_labels: tf.Variable, split_factor: tf.Variable,
                                  first_part: tf.Variable):
    """Filter function for a tfrecord dataset

    Filters the variables X, y and sw by a split factor and labels.

    :param X: List with trainings data
    :param y: List with labels
    :param sw: List with sample weights
    :param use_labels: labels to filter
    :param split_factor: split factor
    :param first_part: A bool value, if true get the first part of the datas else the last part

    :return: Filtered X, y and sw
    """
    label_mask = _select(data=y, allowed=use_labels)
    true_indexes = tf.where(label_mask)
    true_indexes_count = tf.shape(true_indexes)[0]
    split_border = tf.cast(tf.math.floor(split_factor * tf.cast(true_indexes_count, dtype=tf.float32)), dtype=tf.int32)

    selected_indexes = tf.cond(first_part, lambda: true_indexes[:split_border], lambda: true_indexes[split_border:])
    mask = tf.zeros_like(label_mask, dtype=tf.bool)
    update = tf.reshape(tf.ones_like(selected_indexes, dtype=tf.bool), [-1])
    mask = tf.tensor_scatter_nd_update(mask, selected_indexes, update)

    return (tf.boolean_mask(tensor=X, mask=mask), tf.boolean_mask(tensor=y, mask=mask),
            tf.boolean_mask(tensor=sw, mask=mask))


def get_numpy_X(tfr_path: str) -> np.ndarray:
    """Get X as numpy array from a TFRecord file.

    :param tfr_path: Path from the file to read

    :return: Numpy array with data from X
    """
    data = tf.data.TFRecordDataset(filenames=tfr_path).map(map_func=lambda record: tfr_X_parser(record=record))

    return data.as_numpy_iterator().__next__()


def skip_every_x_step(dataset, x_step: int):
    """Skp every X step from a tensorflow dataset

    :param dataset: The parsed dataset
    :param x_step: Every X step will be skipped

    :return: A filtered dataset
    """

    def debug_filter(iteration):
        return tf.equal(iteration % x_step, 0)

    enum_dataset = dataset.enumerate()
    filtered_dataset = enum_dataset.filter(lambda i, _: debug_filter(i))
    return filtered_dataset.map(lambda i, data: data)


def _select(data: tf.Variable, allowed: tf.Variable):
    # get a boolean list for every element in allowed
    use = tf.map_fn(fn=lambda com: tf.equal(x=data, y=com), elems=allowed, fn_output_signature=tf.bool)
    # combine the lists with a logical or
    return tf.cast(tf.math.reduce_sum(input_tensor=tf.cast(use, dtype=tf.int32), axis=0), dtype=tf.bool)
