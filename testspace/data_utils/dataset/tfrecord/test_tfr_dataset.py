import numpy as np
import pytest
import os
from glob import glob

from data_utils.dataset.tfrecord import TFRDatasets
from ..conftest import (
    D1_X_0, D1_X_1, D3_X_0, D3_X_1, Y, WEIGHTS, NAMES_IDX
)
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK

LABELS = [0, 1, 2]
USE_NAMES = ["test_0", "test_1", "test_2", "test_3"]

GET_DATASETS_RANK = [("1d", False, [2, 1]), ("1d", True, [2, 1, 1]),
                     ("3d", False, [4, 1]), ("3d", True, [4, 1, 1])]


def get_test_datasets(test_config, shape: str, with_sw: bool, file_dir: str, batch_size=5):
    test_config.CONFIG_TRAINER[TK.BATCH_SIZE] = batch_size
    test_config.CONFIG_DATALOADER[DLK.D3] = True if shape == "3d" else False
    test_config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS] = with_sw
    tfr_datasets = TFRDatasets(config=test_config)
    paths = glob(os.path.join(file_dir, shape, "shuffled", "*" + TFR_FILE_EXTENSION))
    return tfr_datasets.get_datasets(dataset_paths=paths, train_names=USE_NAMES, valid_names=[], labels=LABELS,
                                     batch_path="")


@pytest.mark.parametrize("shape,with_sw,ranks", GET_DATASETS_RANK)
def test_get_datasets_rank(test_config, tfr_data_dir: str, shape: str, with_sw: bool, ranks: list):
    dataset = get_test_datasets(test_config=test_config, shape=shape, with_sw=with_sw, file_dir=tfr_data_dir)[0]
    for element, rank in zip(dataset.element_spec, ranks):
        assert element.shape.rank == rank


@pytest.mark.parametrize("shape,with_sw,ranks", GET_DATASETS_RANK)
def test_get_datasets_len(test_config, tfr_data_dir: str, shape: str, with_sw: bool, ranks: list):
    dataset = get_test_datasets(test_config=test_config, shape=shape, with_sw=with_sw, file_dir=tfr_data_dir)[0]
    assert len(dataset.element_spec) == len(ranks)


RES_MASK = np.isin(Y, LABELS) * np.isin(NAMES_IDX, [0, 1, 2, 3])
BATCH_X_DATA_1D = np.concatenate((D1_X_0[RES_MASK], D1_X_1[RES_MASK]))
BATCH_X_DATA_3D = np.concatenate((D3_X_0[RES_MASK], D3_X_1[RES_MASK]))
BATCH_Y_DATA = np.concatenate((Y[RES_MASK], Y[RES_MASK]))
BATCH_SW_DATA = np.concatenate((WEIGHTS[RES_MASK], WEIGHTS[RES_MASK]))
GET_DATASET_VALUE = [("1d", False, 5, (BATCH_X_DATA_1D, BATCH_Y_DATA)),
                     ("1d", True, 7, (BATCH_X_DATA_1D, BATCH_Y_DATA, BATCH_SW_DATA)),
                     ("3d", False, 55, (BATCH_X_DATA_3D, BATCH_Y_DATA)),
                     ("3d", True, 33, (BATCH_X_DATA_3D, BATCH_Y_DATA, BATCH_SW_DATA))]


@pytest.mark.parametrize("shape,with_sw,batch_size,results", GET_DATASET_VALUE)
def test_get_datasets_value(test_config, tfr_data_dir: str, shape: str, with_sw: bool, batch_size: int, results: tuple):
    dataset = get_test_datasets(test_config=test_config, shape=shape, with_sw=with_sw, file_dir=tfr_data_dir,
                                batch_size=batch_size)[0]
    max_slice = results[0].shape[0] // batch_size
    length = results[0].shape[0]
    start_slice = 0
    end_slice = batch_size
    for values in dataset:
        for value, result in zip(values, results):
            if start_slice == max_slice:
                end_slice = length
            assert (value.numpy() == result[start_slice:end_slice]).all()
        start_slice = end_slice
        end_slice += batch_size
