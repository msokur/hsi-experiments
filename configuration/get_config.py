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
from configuration.configloader_trainer import read_trainer_config
from configuration.configloader_cv import read_cv_config
from configuration.configloader_dataloader import read_dataloader_config

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def get_config(file_name: str, section: str, main_dir: str = current_dir) -> dict:
    return read_config(file=os.path.join(main_dir, file_name), section=section)


def get_paths(file_name: str, sys_section: str, data_section: str, main_dir: str = current_dir) -> dict:
    return read_path_config(file=os.path.join(main_dir, file_name), system_mode=sys_section, database=data_section)


def get_trainer(file_name: str, section: str, d3: bool, classes: list, main_dir: str = current_dir) -> dict:
    return read_trainer_config(file=os.path.join(main_dir, file_name), section=section, d3=d3, classes=classes)


def get_cv(file_name: str, base_section: str, section: str, main_dir: str = current_dir) -> dict:
    return read_cv_config(file=os.path.join(main_dir, file_name), base_section=base_section, section=section)


def get_dataloader(file_name: str, section: str, main_dir: str = current_dir) -> dict:
    return read_dataloader_config(file=os.path.join(main_dir, file_name), section=section)


# -------- Data Loader
loader_config = "DataLoader.json"
CONFIG_DATALOADER = get_dataloader(file_name=loader_config, section=loader_section)

# --------- Paths
uname = platform.uname()

if "clara" in uname.node:
    system_section = system_section_cluster
else:
    system_section = system_section_local

path_config = "Paths.json"
CONFIG_PATHS = get_paths(file_name=path_config, sys_section=system_section, data_section=database_section)

# --------- Preprocessing
prepro_config = "Preprocessor.json"
CONFIG_PREPROCESSOR = get_config(file_name=prepro_config, section=prepro_section)

# --------- Cross validation
cv_config = "Crossvalidation.json"
CONFIG_CV = get_cv(file_name=cv_config, base_section="BASE", section=cv_section)

# --------- Trainer
trainer_config = "Trainers.json"
CONFIG_TRAINER = get_trainer(file_name=trainer_config, section=trainer_section, d3=CONFIG_DATALOADER["3D"],
                             classes=CONFIG_DATALOADER["LABELS_TO_TRAIN"])

# ----------- DISTRIBUTIONS CHECKING
distro_config = "DistributionsCheck.json"
CONFIG_DISTRIBUTION = get_config(file_name=distro_config, section=distro_section)

# ----------- Augmentation
'''aug_config = "Augmentation.json"
CONFIG_AUG = get_config(file_name=aug_config, section=aug_section)
if CONFIG_AUG["enable"]:
    print("Augmentation is enabled!!!")
    CONFIG_TRAINER["BATCH_SIZE"] = int(CONFIG_TRAINER["BATCH_SIZE"] / CONFIG_AUG[CONFIG_AUG["use"]])'''

# -------- Telegram --------
tg_config = "Telegram.json"
CONFIG_TELEGRAM = get_config(file_name=tg_config, section=tg_section)
CONFIG_TELEGRAM["FILE"] = os.path.join(parent_dir, CONFIG_TELEGRAM["FILE"])
from utils import Telegram

telegram = Telegram(tg_config=CONFIG_TELEGRAM, mode=CONFIG_PATHS["MODE"])
