import os

import pytest


@pytest.fixture
def configuration_dir(main_dir: str) -> str:
    return os.path.join(main_dir, "configuration")


@pytest.fixture
def data_dir(configuration_dir: str) -> str:
    return os.path.join(configuration_dir, "test_data")


@pytest.fixture
def config_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "get_config_data.json")


@pytest.fixture
def paths_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "get_paths_data.json")


@pytest.fixture
def dataloader_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "get_dataloader_data.json")


@pytest.fixture
def paths_data_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "get_paths_data.json")
