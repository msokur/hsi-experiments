import os
import inspect
import platform

import pytest

MAIN_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def main_dir() -> str:
    return MAIN_DIR


@pytest.fixture
def sys_slash() -> str:
    if platform.system() == 'Windows':
        return "\\"
    else:
        return "/"


@pytest.fixture
def test_config():
    from configuration.get_config import (
        get_dataloader,
        get_paths,
        get_config,
        get_cv,
        get_trainer,
        Config,
    )

    config_path = os.path.join(MAIN_DIR, "_test_configs")
    Config.CONFIG_DATALOADER = get_dataloader(file_name="dataloader_data.json", section="DATALOADER",
                                              main_dir=config_path)
    Config.CONFIG_PATHS = get_paths(file_name="paths_data.json", sys_section="SYSTEM", data_section="DATABASE",
                                    main_dir=config_path)
    Config.CONFIG_PREPROCESSOR = get_config(file_name="preprocessor_data.json", section="PREPROCESSOR",
                                            main_dir=config_path)
    Config.CONFIG_CV = get_cv(file_name="cv_data.json", base_section="BASE", section="CV", main_dir=config_path)
    Config.CONFIG_TRAINER = get_trainer(file_name="trainer_data.json", section="NORMAL", d3=True,
                                        classes=[0, 1], main_dir=config_path)
    Config.CONFIG_DISTRIBUTION = get_config(file_name="distro_data.json", section="DISTRIBUTION",
                                            main_dir=config_path)

    return Config


@pytest.fixture
def test_config_tuner(test_config):
    config_path = os.path.join(MAIN_DIR, "_test_configs")
    test_config.CONFIG_TRAINER = test_config.get_trainer(file_name="trainer_data.json", section="NORMAL", d3=True,
                                                         classes=[0, 1], main_dir=config_path)
