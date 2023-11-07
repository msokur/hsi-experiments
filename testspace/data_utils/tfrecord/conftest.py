import pytest
import os


@pytest.fixture
def tfr_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "tfr_file")


@pytest.fixture
def tfr_1d_data_dir(tfr_data_dir) -> str:
    return os.path.join(tfr_data_dir, "1d")


@pytest.fixture
def tfr_3d_data_dir(tfr_data_dir) -> str:
    return os.path.join(tfr_data_dir, "3d")


@pytest.fixture
def tfr_file_name() -> str:
    return "dataset_test.tfrecords"
