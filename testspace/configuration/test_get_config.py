import os

import pytest

from configuration.get_config import get_config, get_paths, get_trainer, get_cv, get_dataloader


def test_set_TF_deterministic():
    assert "TF_DETERMINISTIC_OPS" in os.environ


def test_get_config(config_data_dir: str, base_config_result):
    assert (get_config(file_name="preprocessor_data.json", section="PREPROCESSOR",
                       main_dir=config_data_dir) == base_config_result)


def test_get_paths(config_data_dir: str, path_result: dict):
    assert get_paths(file_name="paths_data.json", sys_section="SYSTEM", data_section="DATABASE",
                     main_dir=config_data_dir) == path_result


@pytest.fixture
def get_trainer_result(trainer_normal_base: dict, model_normal: dict, metric: dict) -> dict:
    result = trainer_normal_base.copy()
    result["MODEL"] = model_normal["3D"]["paper_model"]
    result["CUSTOM_OBJECTS"][0]["metric"] = metric["multi"]["F1_score"]
    result["CUSTOM_OBJECTS_LOAD"].update({"F1_score": metric["multi"]["F1_score"]})
    return result


def test_get_trainer_without_model(config_data_dir: str, get_trainer_result: dict):
    trainer = get_trainer(file_name="trainer_data.json", section="NORMAL", d3=True, classes=[0, 1, 2],
                          main_dir=config_data_dir)
    trainer.pop("MODEL")
    get_trainer_result.pop("MODEL")
    assert trainer == get_trainer_result


def test_get_trainer_with_model(config_data_dir: str, get_trainer_result: dict):
    trainer = get_trainer(file_name="trainer_data.json", section="NORMAL", d3=True, classes=[0, 1, 2],
                          main_dir=config_data_dir)
    trainer_model = trainer.pop("MODEL")
    result_model = get_trainer_result.pop("MODEL")
    assert trainer_model == result_model


def test_get_cv(config_data_dir: str, cv_result):
    assert get_cv(file_name="cv_data.json", base_section="BASE", section="CV", main_dir=config_data_dir) == cv_result


def test_get_dataloader(config_data_dir: str, dataloader_result):
    assert get_dataloader(file_name="dataloader_data.json", section="DATALOADER",
                          main_dir=config_data_dir) == dataloader_result
