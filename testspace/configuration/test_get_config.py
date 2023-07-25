import os

from configuration.get_config import get_config, get_paths, get_trainer, get_cv, get_dataloader


def test_set_TF_deterministic():
    assert "TF_DETERMINISTIC_OPS" in os.environ


def test_get_config(config_data_name: str, data_dir: str, base_config_result):
    assert get_config(file_name=config_data_name, section="TEST", main_dir=data_dir) == base_config_result


def test_get_paths(paths_data_name: str, data_dir: str, path_result: dict):
    assert get_paths(file_name=paths_data_name, sys_section="SYSTEM", data_section="DATABASE",
                     main_dir=data_dir) == path_result


def test_get_trainer():
    assert False


def test_get_cv(cv_data_name: str, data_dir: str, cv_result):
    assert get_cv(file_name=cv_data_name, base_section="BASE", section="CV", main_dir=data_dir) == cv_result


def test_get_dataloader(dataloader_data_name: str, data_dir: str, dataloader_result):
    assert get_dataloader(file_name=dataloader_data_name, section="DATALOADER",
                          main_dir=data_dir) == dataloader_result
