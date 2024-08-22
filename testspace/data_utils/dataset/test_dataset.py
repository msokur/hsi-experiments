import numpy as np
import pytest
import os

from shutil import rmtree
from glob import glob
from data_utils.dataset import TFRDatasets, GeneratorDatasets
from data_utils.data_storage import DataStorageZARR, DataStorageNPZ
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK

from .conftest import (
    USE_NAMES,
    LABELS,
    BATCH_X_DATA_1D,
    BATCH_X_DATA_3D,
    BATCH_Y_DATA,
    BATCH_SW_DATA,
)


@pytest.fixture
def _delete_batches(data_dir: str):
    yield
    path = os.path.join(data_dir, "test_batches")
    if os.path.exists(path=path):
        rmtree(path=path)


def get_test_datasets(test_config, data_typ: str, shape: str, with_sw: bool, file_dir: str, batch_size: int,
                      valid_names=USE_NAMES):
    test_config.CONFIG_TRAINER[TK.BATCH_SIZE] = batch_size
    test_config.CONFIG_DATALOADER[DLK.D3] = True if shape == "3d" else False
    test_config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS] = with_sw

    sh_path = os.path.join(file_dir, f"{data_typ}_file", shape, "shuffled")
    if data_typ == "tfr":
        datasets = TFRDatasets(config=test_config)
        paths = glob(os.path.join(sh_path, "*" + TFR_FILE_EXTENSION))
    elif data_typ == "npz":
        npz_storage = DataStorageNPZ()
        datasets = GeneratorDatasets(config=test_config,
                                     data_storage=npz_storage)
        paths = npz_storage.get_paths(storage_path=sh_path)
    else:
        zarr_storage = DataStorageZARR()
        datasets = GeneratorDatasets(config=test_config,
                                     data_storage=zarr_storage)
        paths = zarr_storage.get_paths(storage_path=sh_path)

    return datasets.get_datasets(dataset_paths=paths,
                                 train_names=USE_NAMES,
                                 valid_names=valid_names,
                                 labels=LABELS,
                                 batch_path=os.path.join(file_dir, "test_batches"))


GET_DATASET_VALUE = [("tfr", "1d", False, 5, (BATCH_X_DATA_1D, BATCH_Y_DATA)),
                     ("tfr", "1d", True, 7, (BATCH_X_DATA_1D, BATCH_Y_DATA, BATCH_SW_DATA)),
                     ("tfr", "3d", False, 55, (BATCH_X_DATA_3D, BATCH_Y_DATA)),
                     ("tfr", "3d", True, 119, (BATCH_X_DATA_3D, BATCH_Y_DATA, BATCH_SW_DATA)),
                     ("npz", "1d", False, 5, (BATCH_X_DATA_1D, BATCH_Y_DATA)),
                     ("npz", "1d", True, 7, (BATCH_X_DATA_1D, BATCH_Y_DATA, BATCH_SW_DATA)),
                     ("npz", "3d", False, 120, (BATCH_X_DATA_3D, BATCH_Y_DATA)),
                     ("npz", "3d", True, 33, (BATCH_X_DATA_3D, BATCH_Y_DATA, BATCH_SW_DATA)),
                     ("zarr", "1d", False, 70, (BATCH_X_DATA_1D, BATCH_Y_DATA)),
                     ("zarr", "1d", True, 13, (BATCH_X_DATA_1D, BATCH_Y_DATA, BATCH_SW_DATA)),
                     ("zarr", "3d", False, 24, (BATCH_X_DATA_3D, BATCH_Y_DATA)),
                     ("zarr", "3d", True, 117, (BATCH_X_DATA_3D, BATCH_Y_DATA, BATCH_SW_DATA))]


@pytest.mark.parametrize("data_type,shape,with_sw,batch_size,results", GET_DATASET_VALUE)
def test_get_datasets_value(_delete_batches, test_config, data_dir: str, data_type: str, shape: str, with_sw: bool,
                            batch_size: int, results: tuple):
    dataset = get_test_datasets(test_config=test_config,
                                data_typ=data_type,
                                shape=shape,
                                with_sw=with_sw,
                                file_dir=data_dir,
                                batch_size=batch_size)[0]
    start_slice = 0
    end_slice = batch_size
    for values in dataset:
        for value, result in zip(values, results):
            if not np.all(value.numpy() == result[start_slice:end_slice]):
                print(f"---- {start_slice} -- {end_slice} ----")
                print(f"dataset:\n{value.numpy()}\n")
                print(f"result:\n{result[start_slice:end_slice]}\n")
            assert np.all(value.numpy() == result[start_slice:end_slice])
        start_slice = end_slice
        end_slice += batch_size


GET_DATASET_VALUE_ERROR = [("tfr", "1d", 105, "validation"),
                           ("tfr", "3d", 205, "training"),
                           ("npz", "3d", 105, "validation"),
                           ("zarr", "1d", 205, "training")]


@pytest.mark.parametrize("data_type,shape,bach_size,dataset_typ", GET_DATASET_VALUE_ERROR)
def test_get_datasets_value_error(_delete_batches, test_config, data_dir: str, data_type: str, shape: str,
                                  bach_size: int, dataset_typ: str):
    with pytest.raises(ValueError,
                       match=f"There to less data for a {dataset_typ} dataset. Maybe lower the batch size!"):
        get_test_datasets(test_config=test_config,
                          data_typ=data_type,
                          shape=shape,
                          with_sw=False,
                          file_dir=data_dir,
                          batch_size=bach_size,
                          valid_names=USE_NAMES if bach_size > 200 else USE_NAMES[0:2])


GET_DATASET_NO_SH_FILES_DATA = ["tfr", "npz", "zarr"]


@pytest.mark.parametrize("data_type", GET_DATASET_NO_SH_FILES_DATA)
def test_get_datasets_no_shuffle_files_error(test_config, data_type: str):
    with pytest.raises(ValueError, match="No shuffle file to create a dataset. Check your path configs!"):
        get_test_datasets(test_config=test_config,
                          data_typ=data_type,
                          shape="1d",
                          with_sw=False,
                          file_dir="",
                          batch_size=0)
