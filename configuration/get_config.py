import inspect
import os
import sys
import platform

from utils import Telegram

from configuration.load_config import read_config, read_path_config, read_trainer_config, read_cv_config

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

WITHOUT_RANDOMNESS = True
if WITHOUT_RANDOMNESS:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def get_config(file_name: str, section: str) -> dict:
    return read_config(file=os.path.join(current_dir, file_name), section=section)


def get_paths(file_name: str, sys_section: str, data_section: str) -> dict:
    return read_path_config(file=os.path.join(current_dir, file_name), system_mode=sys_section, database=data_section)


def get_trainer(file_name: str, section: str, d3: bool, classes: list) -> dict:
    return read_trainer_config(file=os.path.join(current_dir, file_name), section=section, d3=d3, classes=classes)


def get_cv(file_name: str, section: str) -> dict:
    return read_cv_config(file=os.path.join(current_dir, file_name), section=section)


# -------- Data Loader
loader_config = "DataLoader.json"
loader_section = "HNO"
DATALOADER = get_config(file_name=loader_config, section=loader_section)

# --------- Paths
uname = platform.uname()

if "clara" in uname.node:
    system_section = "Cluster_Benny"
else:
    system_section = "Win_Benny"

path_config = "Paths.json"
database_section = "HNO_Database"
PATHS = get_paths(file_name=path_config, sys_section=system_section, data_section=database_section)

# --------- Preprocessing
prepro_config = "Preprocessor.json"
prepro_section = "HNO"
PREPRO = get_config(file_name=prepro_config, section=prepro_section)

# --------- Cross validation
cv_config = "Crossvalidation.json"
cv_section = "CV_HNO_NORMAL"
CV = get_config(file_name=cv_config, section=cv_section)

# --------- Trainer
trainer_config = "Trainers.json"
trainer_section = "HNO"
TRAINER = get_trainer(file_name=trainer_config, section=trainer_section, d3=DATALOADER["3D"],
                      classes=DATALOADER["LABELS_TO_TRAIN"])

# ----------- DISTRIBUTIONS CHECKING
distro_config = "DistributionsCheck.json"
distro_section = "Default"
DISTRO = get_config(file_name=distro_config, section=distro_section)

# ----------- Augmentation
aug_config = "Augmentation.json"
aug_section = "Default"
AUG = get_config(file_name=aug_config, section=aug_section)
if AUG["enable"]:
    print("Augmentation is enabled!!!")
    TRAINER["BATCH_SIZE"] = int(TRAINER["BATCH_SIZE"] / AUG[AUG["use"]])


# -------- Telegram --------
tg_config = "Telegram.json"
tg_section = "Benny"
TG_CONF = get_config(file_name=tg_config, section=tg_section)
TG_CONF["FILE"] = os.path.join(parent_dir, TG_CONF["FILE"])
telegram = Telegram(tg_config=TG_CONF, mode=PATHS["MODE"])
