from glob import glob
from tensorflow import TensorSpec
import numpy as np
import os

import pytest

from data_utils.data_archive import DataArchiveZARR, DataArchiveNPZ, DataArchive
from data_utils.batches.data_generator import DataGenerator
from testspace.data_utils.conftest import (
    DATA_1D_X_0, DATA_1D_X_1, DATA_3D_X_0, DATA_3D_X_1, DATA_y_0, DATA_y_1, DATA_WEIGHTS_0, DATA_WEIGHTS_1,
)

X_NAME = "X"
y_NAME = "y"
WEIGHTS_NAME = "weights"

D1_X_y_0 = (DATA_1D_X_0, DATA_y_0)
D1_X_y_w_0 = D1_X_y_0 + (DATA_WEIGHTS_0,)
D3_X_y_0 = (DATA_3D_X_0, DATA_y_0)
D3_X_y_w_0 = D3_X_y_0 + (DATA_WEIGHTS_0,)

D1_X_y_1 = (DATA_1D_X_1, DATA_y_1)
D1_X_y_w_1 = D1_X_y_1 + (DATA_WEIGHTS_1,)
D3_X_y_1 = (DATA_3D_X_1, DATA_y_1)
D3_X_y_w_1 = D3_X_y_1 + (DATA_WEIGHTS_1,)

X_1D_TF_SPEC = TensorSpec(shape=(None, 10), dtype=np.int32, name=X_NAME)
X_3D_TF_SPEC = TensorSpec(shape=(None, 3, 3, 10), dtype=np.int32, name=X_NAME)
y_TF_SPEC = TensorSpec(shape=(None,), dtype=np.int32, name=y_NAME)
WEIGHT_TF_SPEC = TensorSpec(shape=(None,), dtype=np.float64, name=WEIGHTS_NAME)


def test___len__(zarr_3d_data_dir: str):
    batch_paths = glob(os.path.join(zarr_3d_data_dir, "data_test_*"))
    data_gen = DataGenerator(data_archive=DataArchiveZARR(), batch_paths=batch_paths, X_name=X_NAME, y_name=y_NAME,
                             weights_name=WEIGHTS_NAME, with_sample_weights=False)
    assert data_gen.__len__() == 2


GETITEM_TEST_DATA = [(DataArchiveZARR(), "zarr", "1d", 0, False, D1_X_y_0),
                     (DataArchiveZARR(), "zarr", "3d", 1, True, D3_X_y_w_1),
                     (DataArchiveNPZ(), "npz", "1d", 1, False, D1_X_y_1),
                     (DataArchiveNPZ(), "npz", "3d", 0, True, D3_X_y_w_0)]


@pytest.mark.parametrize("data_archive,typ,patch,idx,with_weights,results", GETITEM_TEST_DATA)
def test___getitem__(data_dir: str, data_archive: DataArchive, typ: str, patch: str, idx: int, with_weights: bool,
                     results: tuple):
    batch_paths = sorted(glob(os.path.join(data_dir, f"{typ}_file", patch, "data_test_*")))
    print(batch_paths)
    data_gen = DataGenerator(data_archive=data_archive, batch_paths=batch_paths, X_name=X_NAME, y_name=y_NAME,
                             weights_name=WEIGHTS_NAME, with_sample_weights=with_weights)
    data = data_gen.__getitem__(idx=idx)
    for d, r in zip(data, results):
        assert (d == r).all()


CALL_TEST_DATA = [(DataArchiveZARR(), "zarr", "1d", True, [D1_X_y_w_0, D1_X_y_w_1]),
                  (DataArchiveZARR(), "zarr", "3d", False, [D3_X_y_0, D3_X_y_1]),
                  (DataArchiveNPZ(), "npz", "1d", True, [D1_X_y_w_0, D1_X_y_w_1]),
                  (DataArchiveNPZ(), "npz", "3d", False, [D3_X_y_0, D3_X_y_1])]


@pytest.mark.parametrize("data_archive,typ,patch,with_weights,results", CALL_TEST_DATA)
def test___call__(data_dir: str, data_archive: DataArchive, typ: str, patch: str, with_weights: bool, results: list):
    batch_paths = sorted(glob(os.path.join(data_dir, f"{typ}_file", patch, "data_test_*")))
    data_gen = DataGenerator(data_archive=data_archive, batch_paths=batch_paths, X_name=X_NAME, y_name=y_NAME,
                             weights_name=WEIGHTS_NAME, with_sample_weights=with_weights)
    for datas, result in zip(data_gen.__call__(), results):
        for data, res in zip(datas, result):
            assert (data == res).all()


SIGNATURE_TEST_DATA = [(DataArchiveZARR(), "zarr", "1d", True, (X_1D_TF_SPEC, y_TF_SPEC, WEIGHT_TF_SPEC)),
                       (DataArchiveZARR(), "zarr", "3d", False, (X_3D_TF_SPEC, y_TF_SPEC)),
                       (DataArchiveNPZ(), "npz", "1d", False, (X_1D_TF_SPEC, y_TF_SPEC)),
                       (DataArchiveNPZ(), "npz", "3d", True, (X_3D_TF_SPEC, y_TF_SPEC, WEIGHT_TF_SPEC))]


@pytest.mark.parametrize("data_archive,typ,patch,with_weights,results", SIGNATURE_TEST_DATA)
def test_get_output_signature(data_dir: str, data_archive: DataArchive, typ: str, patch: str, with_weights: bool,
                              results: tuple):
    batch_paths = glob(os.path.join(data_dir, f"{typ}_file", patch, "data_test_*"))
    data_gen = DataGenerator(data_archive=data_archive, batch_paths=batch_paths, X_name=X_NAME, y_name=y_NAME,
                             weights_name=WEIGHTS_NAME, with_sample_weights=with_weights)
    assert data_gen.get_output_signature() == results
