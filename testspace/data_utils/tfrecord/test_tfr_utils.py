import pytest
import tensorflow as tf
import numpy as np

from data_utils.tfrecord.tfr_utils import get_features, tfr_parser, get_class_weights


SIZE_1D = (10, 9)
SIZE_3D = (10, 3, 3, 9)
X_1D = np.arange(90).reshape(SIZE_1D).astype(np.float32)
X_3D = np.arange(810).reshape(SIZE_3D).astype(np.float32)
Y_1D = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
Y_2D = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1]])
WEIGHTS = np.array([100, 200, 300, 400, 100, 200, 300, 400, 100, 200])


def test_get_features():
    assert False


def test_tfr_paser():
    assert False


GET_CLASS_WEIGHTS_DATA = [(np.array([0, 1, 2, 3]), {0: 0.83, 1: 0.83, 2: 1.25, 3: 1.25}),
                          (np.array([0, 2, 3]), {0: 0.77, 2: 1.16, 3: 1.16})]


@pytest.mark.parametrize("labels,result", GET_CLASS_WEIGHTS_DATA)
def test_get_class_weights(labels: np.ndarray, result: dict):
    dataset = tf.data.Dataset.from_tensor_slices(([X_3D], [Y_1D]))
    weights = get_class_weights(dataset=dataset, labels=labels)
    print(weights)
    print(result)
    for (k_ds, v_ds), (k_r, v_r) in zip(weights.items(), result.items()):
        assert k_ds == k_r and v_ds == pytest.approx(v_ds, rel=1e-2)
