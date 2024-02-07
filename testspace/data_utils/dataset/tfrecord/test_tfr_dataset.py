import numpy as np
import pytest
import os
from glob import glob

from data_utils.dataset.tfrecord import TFRDatasets
from testspace.data_utils.dataset.tfrecord.conftest import (
    TF_DATA_1D_X_0, TF_DATA_3D_X_0, TF_DATA_y_0, TF_DATA_WEIGHTS_0, TF_NAMES_IDX
)
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)


BATCH_SIZE = 5
LABELS = [0, 1, 2]
USE_NAMES = ["test_0", "test_1", "test_2", "test_3"]

GET_DATASETS_RANK = [("1d", False, [2, 1]), ("1d", True, [2, 1, 1]),
                     ("3d", False, [4, 1]), ("3d", True, [4, 1, 1])]


def get_test_datasets(shape: str, with_sw: bool, file_dir: str):
    tfr_datasets = TFRDatasets(batch_size=BATCH_SIZE, d3=True if shape == "3d" else False, with_sample_weights=with_sw)
    paths = glob(os.path.join(file_dir, shape, "*" + TFR_FILE_EXTENSION))
    return tfr_datasets.get_datasets(dataset_paths=paths, train_names=USE_NAMES, valid_names=[], labels=LABELS,
                                     batch_path="")


@pytest.mark.parametrize("shape,with_sw,ranks", GET_DATASETS_RANK)
def test_get_datasets_rank(tfr_data_dir: str, shape: str, with_sw: bool, ranks: list):
    dataset = get_test_datasets(shape=shape, with_sw=with_sw, file_dir=tfr_data_dir)[0]
    for element, rank in zip(dataset.element_spec, ranks):
        assert element.shape.rank == rank


@pytest.mark.parametrize("shape,with_sw,ranks", GET_DATASETS_RANK)
def test_get_datasets_len(tfr_data_dir: str, shape: str, with_sw: bool, ranks: list):
    dataset = get_test_datasets(shape=shape, with_sw=with_sw, file_dir=tfr_data_dir)[0]
    assert len(dataset.element_spec) == len(ranks)


RES_MASK = np.isin(TF_DATA_y_0, LABELS) * np.isin(TF_NAMES_IDX, [0, 1, 2, 3])
GET_DATASET_VALUE = [("1d", False, (TF_DATA_1D_X_0[RES_MASK], TF_DATA_y_0[RES_MASK])),
                     ("1d", True, (TF_DATA_1D_X_0[RES_MASK], TF_DATA_y_0[RES_MASK], TF_DATA_WEIGHTS_0[RES_MASK])),
                     ("3d", False, (TF_DATA_3D_X_0[RES_MASK], TF_DATA_y_0[RES_MASK])),
                     ("3d", True, (TF_DATA_3D_X_0[RES_MASK], TF_DATA_y_0[RES_MASK], TF_DATA_WEIGHTS_0[RES_MASK]))]


@pytest.mark.parametrize("shape,with_sw,results", GET_DATASET_VALUE)
def test_get_datasets_value(tfr_data_dir: str, shape: str, with_sw: bool, results: tuple):
    dataset = get_test_datasets(shape=shape, with_sw=with_sw, file_dir=tfr_data_dir)[0]
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
