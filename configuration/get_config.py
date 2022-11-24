import inspect
import os
import sys

from utils import Telegram

from configuration.load_config import read_config, read_path_config
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
'''sys.path.insert(1, os.path.join(parent_dir, "util"))
sys.path.insert(2, os.path.join(parent_dir, "data_utils"))
sys.path.insert(3, os.path.join(parent_dir, os.path.join("'data_utils', 'data_loaders'")))
sys.path.insert(4, os.path.join(parent_dir, os.path.join("'data_utils', 'configuration'")))
sys.path.insert(5, os.path.join(parent_dir, "models"))
sys.path.insert(6, os.path.join(parent_dir, "trainers"))
sys.path.insert(7, os.path.join(parent_dir, "cross_validator"))'''

# -------- Data Loader
loader_config = os.path.join(current_dir, "DataLoader.json")
loader_section = "HNO"
DATALOADER = read_config(file=loader_config, section=loader_section)

# --------- Paths
path_config = os.path.join(current_dir, "Paths.json")
system_section = "Win_Benny"
database_section = "HNO_Database"
PATHS = read_path_config(file=path_config, system_mode=system_section, database=database_section)

# --------- Preprocessing
prepro_config = os.path.join(current_dir, "Preprocessor.json")
prepro_section = "HNO"
PREPRO = read_config(file=prepro_config, section=prepro_section)

# ----------- DISTRIBUTIONS CHECKING
distro_config = os.path.join(current_dir, "DistributionsCheck.json")
distro_section = "Default"
DISTRO = read_config(file=distro_config, section=distro_section)

# -------- Telegram --------
tg_config = os.path.join(current_dir, "Telegram.json")
tg_section = "Benny"
TG_CONF = read_config(file=tg_config, section=tg_section)
TG_CONF["FILE"] = os.path.join(parent_dir, TG_CONF["FILE"])
telegram = Telegram(tg_config=TG_CONF, mode=PATHS["MODE"])
