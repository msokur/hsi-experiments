import pytest
import os


@pytest.fixture
def mat_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "mat_file")


@pytest.fixture
def zarr_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "zarr_file")
