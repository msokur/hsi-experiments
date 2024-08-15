import inspect
import os
import sys
import platform

WITHOUT_RANDOMNESS = True
if WITHOUT_RANDOMNESS:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
else:
    os.environ['TF_DETERMINISTIC_OPS'] = '0'

# to mute tensorflow logs
# 0 - all logs will be shown
# 1 - info logs muted
# 2 - info, warning logs muted
# 3 - all logs muted
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# change here the name of the .py file for configuration import
from configuration.meta_configs.Benny import *

from configuration.configloader_base import read_config
from configuration.configloader_paths import read_path_config
from configuration.configloader_cv import read_cv_config
from configuration.configloader_dataloader import read_dataloader_config
from utils import Telegram

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def get_config(file_name: str, section: str, main_dir: str = current_dir) -> dict:
    return read_config(file=os.path.join(main_dir, file_name),
                       section=section)


def get_paths(file_name: str, sys_section: str, data_section: str, main_dir: str = current_dir) -> dict:
    return read_path_config(file=os.path.join(main_dir, file_name),
                            system_mode=sys_section,
                            database=data_section)


def get_trainer(file_name: str, section: str, d3: bool, classes: list, main_dir: str = current_dir) -> dict:
    from configuration.configloader_trainer import read_trainer_config
    return read_trainer_config(file=os.path.join(main_dir, file_name),
                               section=section,
                               d3=d3,
                               classes=classes)


def get_cv(file_name: str, base_section: str, section: str, main_dir: str = current_dir) -> dict:
    return read_cv_config(file=os.path.join(main_dir, file_name),
                          base_section=base_section,
                          section=section)


def get_dataloader(file_name: str, section: str, main_dir: str = current_dir) -> dict:
    return read_dataloader_config(file=os.path.join(main_dir, file_name),
                                  section=section)


class PreprocessorConfig:
    def __init__(self):
        # -------- Data Loader
        self.CONFIG_DATALOADER = get_dataloader(file_name=loader_config_file,
                                                section=loader_section)

        # --------- Paths
        uname = platform.uname()

        if "clara" in uname.node:
            system_section = system_section_cluster
            self.CLUSTER = True
        else:
            system_section = system_section_local
            self.CLUSTER = False

        self.CONFIG_PATHS = get_paths(file_name=path_config_file,
                                      sys_section=system_section,
                                      data_section=database_section)

        # --------- Preprocessing
        self.CONFIG_PREPROCESSOR = get_config(file_name=prepro_config_file,
                                              section=prepro_section)

        # ----------- DISTRIBUTIONS CHECKING
        self.CONFIG_DISTRIBUTION = get_config(file_name=distro_config_file,
                                              section=distro_section)

        # -------- Telegram --------
        self.CONFIG_TELEGRAM = get_config(file_name=tg_config_file,
                                          section=tg_section)
        self.CONFIG_TELEGRAM["FILE"] = os.path.join(parent_dir, self.CONFIG_TELEGRAM["FILE"])

        self.telegram = Telegram(tg_config=self.CONFIG_TELEGRAM,
                                 mode=self.CONFIG_PATHS["MODE"])


class CVConfig(PreprocessorConfig):
    def __init__(self):
        super().__init__()

        # --------- Cross validation
        self.CONFIG_CV = get_cv(file_name=cv_config_file,
                                base_section="BASE",
                                section=cv_section)

        # --------- Trainer
        self.CONFIG_TRAINER = get_trainer(file_name=trainer_config_file,
                                          section=trainer_section,
                                          d3=self.CONFIG_DATALOADER["3D"],
                                          classes=self.CONFIG_DATALOADER["LABELS_TO_TRAIN"])
