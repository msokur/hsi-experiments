import numpy as np
import pytest
import os

from data_utils.tfrecord import TFRDatasets
from testspace.data_utils.conftest import (
    DATA_1D_X_0, DATA_1D_X_1, DATA_3D_X_0, DATA_3D_X_1, DATA_y_0, DATA_y_1, DATA_WEIGHTS_0, DATA_WEIGHTS_1,
)


BATCH_SIZE = 5

GET_DATASETS_RANK = [("1d", False, [2, 1]), ("1d", True, [2, 1, 1]),
                     ("3d", False, [4, 1]), ("3d", True, [4, 1, 1])]


def get_test_datasets(shape: str, with_sw: bool, file_dir: str, file_name: str):
    tfr_datasets = TFRDatasets(batch_size=BATCH_SIZE, d3=True if shape == "3d" else False, with_sample_weights=with_sw)
    file = os.path.join(file_dir, shape, file_name)
    return tfr_datasets.get_datasets(train_tfr_file=file, valid_tfr_file=file)


@pytest.mark.parametrize("shape,with_sw,ranks", GET_DATASETS_RANK)
def test_get_datasets_rank(tfr_data_dir: str, tfr_file_name: str, shape: str, with_sw: bool, ranks: list):
    dataset = get_test_datasets(shape=shape, with_sw=with_sw, file_dir=tfr_data_dir, file_name=tfr_file_name)[0]
    for element, rank in zip(dataset.element_spec, ranks):
        assert element.shape.rank == rank


X_1D_CON = np.concatenate((DATA_1D_X_0, DATA_1D_X_1), axis=0, dtype=np.float32)
X_3D_CON = np.concatenate((DATA_3D_X_0, DATA_3D_X_1), axis=0, dtype=np.float32)
Y_CON = np.concatenate((DATA_y_0, DATA_y_1), axis=0, dtype=np.float32)
WEIGHTS_CON = np.concatenate((DATA_WEIGHTS_0, DATA_WEIGHTS_1), axis=0, dtype=np.float32)

GET_DATASET_VALUE = [("1d", False, (X_1D_CON, Y_CON)), ("1d", True, (X_1D_CON, Y_CON, WEIGHTS_CON)),
                     ("3d", False, (X_3D_CON, Y_CON)), ("3d", True, (X_3D_CON, Y_CON, WEIGHTS_CON))]


@pytest.mark.parametrize("shape,with_sw,results", GET_DATASET_VALUE)
def test_get_datasets_value(tfr_data_dir: str, tfr_file_name: str, shape: str, with_sw: bool, results: tuple):
    dataset = get_test_datasets(shape=shape, with_sw=with_sw, file_dir=tfr_data_dir, file_name=tfr_file_name)[0]
    max_slice = results[0].shape[0] // BATCH_SIZE
    length = results[0].shape[0]
    start_slice = 0
    end_slice = BATCH_SIZE
    for values in dataset:
        for value, result in zip(values, results):
            if start_slice == max_slice:
                end_slice = length
            assert (value.numpy() == result[start_slice:end_slice]).all()
        start_slice = end_slice
        end_slice += BATCH_SIZE
