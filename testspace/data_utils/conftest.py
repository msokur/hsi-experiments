import pytest
import os


@pytest.fixture
def package_dir(main_dir: str) -> str:
    return os.path.join(main_dir, "data_utils")


@pytest.fixture
def data_dir(package_dir: str) -> str:
    return os.path.join(package_dir, "test_data")


@pytest.fixture
def dat_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "dat_file")


@pytest.fixture
def npz_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "npz_file")


@pytest.fixture
def npz_1d_data_dir(npz_data_dir: str) -> str:
    return os.path.join(npz_data_dir, "1d")


@pytest.fixture
def npz_3d_data_dir(npz_data_dir: str) -> str:
    return os.path.join(npz_data_dir, "3d")


@pytest.fixture
def zarr_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "zarr_file")


@pytest.fixture
def zarr_1d_data_dir(zarr_data_dir: str) -> str:
    return os.path.join(zarr_data_dir, "1d")


@pytest.fixture
def zarr_3d_data_dir(zarr_data_dir: str) -> str:
    return os.path.join(zarr_data_dir, "3d")


@pytest.fixture
def tfr_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "tfr_file")


@pytest.fixture
def tfr_1d_data_dir(tfr_data_dir) -> str:
    return os.path.join(tfr_data_dir, "1d")


@pytest.fixture
def tfr_3d_data_dir(tfr_data_dir) -> str:
    return os.path.join(tfr_data_dir, "3d")


# --- datas ---
import numpy as np

SPEC = 10
SAMPLES = 12

DATA_1D_X_0 = np.arange(start=0, stop=SAMPLES * SPEC, step=1).reshape((SAMPLES, SPEC))
DATA_1D_X_1 = np.arange(start=SAMPLES * SPEC, stop=SAMPLES * SPEC * 2, step=1).reshape((SAMPLES, SPEC))

DATA_3D_X_0 = np.array([list(x) * 9 for x in DATA_1D_X_0]).reshape((SAMPLES, 3, 3, SPEC))
DATA_3D_X_1 = np.array([list(x) * 9 for x in DATA_1D_X_1]).reshape((SAMPLES, 3, 3, SPEC))

DATA_y_0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
DATA_y_1 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

DATA_WEIGHTS_0 = np.array([4.] * DATA_y_0.shape[0])
DATA_WEIGHTS_1 = np.array([2.] * DATA_y_1.shape[0])

DATA_i = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]])

DATA_1D_0 = {"X": DATA_1D_X_0, "indexes_in_datacube": DATA_i, "weights": DATA_WEIGHTS_0, "y": DATA_y_0}
DATA_1D_1 = {"X": DATA_1D_X_1, "indexes_in_datacube": DATA_i, "weights": DATA_WEIGHTS_1, "y": DATA_y_1}

DATA_3D_0 = {"X": DATA_3D_X_0, "indexes_in_datacube": DATA_i, "weights": DATA_WEIGHTS_0, "y": DATA_y_0}
DATA_3D_1 = {"X": DATA_3D_X_1, "indexes_in_datacube": DATA_i, "weights": DATA_WEIGHTS_1, "y": DATA_y_1}
