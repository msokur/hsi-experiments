import pytest
import os


@pytest.fixture
def package_dir(main_dir: str) -> str:
    return os.path.join(main_dir, "data_utils")


@pytest.fixture
def data_dir(package_dir: str) -> str:
    return os.path.join(package_dir, "data_loaders", "test_data")


@pytest.fixture
def dat_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "dat_file")


@pytest.fixture
def npz_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "npz_file")


@pytest.fixture
def zarr_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "zarr_file")
