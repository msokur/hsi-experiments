import pytest
import platform

import os
import inspect

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(file_dir, "test_data")

from configuration.get_config import get_config, get_paths, get_trainer, get_cv, get_dataloader

if platform.system() == 'Windows':
    sys = "\\"
else:
    sys = "/"


def test_set_TF_deterministic():
    assert "TF_DETERMINISTIC_OPS" in os.environ


GET_CONFIG_DATA = {"EXECUTION_FLAGS": {"LOAD_DATA_WITH_DATALOADER": True,
                                       "ADD_SAMPLE_WEIGHTS": True,
                                       "SCALE": True,
                                       "SHUFFLE": True},
                   "DICT_NAMES": ["NAME_1", "NAME_2", "NAME_3"],
                   "PILES_NUMBER": 100,
                   "WEIGHT_FILENAME": "weights.weights",
                   "FILES_TO_COPY": ["test_file.py"],
                   "NORMALIZATION_TYPE": "svn",
                   "SCALER_FILE": "scaler.scaler",
                   "SCALER_PATH": "scaler_path"
                   }


def test_get_config():
    configuration = get_config(file_name="get_config_data.json", section="TEST", main_dir=data_dir)
    assert configuration == GET_CONFIG_DATA


GET_PATHS_DATA = {"CHECKPOINT_PATH": "checkpoints",
                  "MODEL_PATH": "model",
                  "MODE": "WITH_GPU",
                  "PREFIX": "/work/users/xyz",
                  "MODEL_NAME_PATHS": ["/home/sc.uni-leipzig.de/xyz/hsi-experiments-BA/logs"],
                  "RAW_NPZ_PATH": f"/work/users/xyz{sys}data_3d{sys}gastric{sys}3x3",
                  "RAW_SOURCE_PATH": f"/work/users/xyz{sys}Gastric",
                  "TEST_NPZ_PATH": f"/work/users/xyz{sys}data_3d{sys}hno{sys}3x3",
                  "SHUFFLED_PATH": f"/work/users/xyz{sys}data_3d{sys}gastric{sys}3x3{sys}shuffled",
                  "BATCHED_PATH": f"/work/users/xyz{sys}data_3d{sys}gastric{sys}3x3{sys}batch_sized",
                  "MASK_PATH": f"/work/users/xyz{sys}Gastric{sys}annotation",
                  "SYSTEM_PATHS_DELIMITER": sys}


def test_get_paths():
    configuration = get_paths(file_name="get_paths_data.json", sys_section="SYSTEM", data_section="DATABASE",
                              main_dir=data_dir)
    assert configuration == GET_PATHS_DATA


def test_get_trainer():
    assert False


GET_CV_DATA = {"EXECUTION_FLAGS": {"cross_validation": True,
                                   "evaluation": True},
               "FIRST_SPLIT": 0,
               "HOW_MANY_VALID_EXCLUDE": 1,
               "CHOOSE_EXCLUDED_VALID": "by_class",
               "USE_ALL_LABELS": False,
               "SAVE_PREDICTION": True,
               "SAVE_CURVES": False,
               "TYPE": "normal",
               "NAME": "CV_NAME",
               "DATABASE_ABBREVIATION": "DATABASE_NAME",
               "RESTORE_VALID_PATH": "/home/sc.uni-leipzig.de/xyz/hsi-experiments-BA/logs/Experiment",
               "RESTORE_VALID_SEQUENCE": "",
               "GET_CHECKPOINT_FROM_VALID": True
               }


def test_get_cv():
    configuration = get_cv(file_name="get_cv_data.json", base_section="BASE", section="CV", main_dir=data_dir)
    assert configuration == GET_CV_DATA


GET_DATALOADER_DATA = {"TYPE": "normal",
                       "FILE_EXTENSION": ".dat",
                       "3D": True,
                       "3D_SIZE": [3, 3],
                       "FIRST_NM": 8,
                       "LAST_NM": 100,
                       "WAVE_AREA": 100,
                       "LABELS_TO_TRAIN": [0, 1],
                       "NAME_SPLIT": "_SpecCube",
                       "MASK_DIFF": ["_SpecCube.dat", ".png"],
                       "LABELS_FILENAME": "labels.labels",
                       "CONTAMINATION_FILENAME": "contamination.csv",
                       "SMOOTHING_TYPE": "median_filter",
                       "SMOOTHING_VALUE": 5,
                       "BORDER_CONFIG": {
                           "enable": False,
                           "methode": "detect_core",
                           "depth": 5,
                           "axis": [],
                           "not_used_labels": []
                       },
                       "SPLIT_PATHS_BY": "Files",
                       "CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST": 1,
                       "WITH_BACKGROUND_EXTRACTION": False,
                       "MASK_COLOR": {0: [[255, 255, 0]], 1: [[0, 0, 255]], 2: [[255, 0, 0]]},
                       "TISSUE_LABELS": {0: "Nerve", 1: "Tumor", 2: "Parotis"},
                       "PLOT_COLORS": {0: "yellow", 1: "blue", 2: "red"},
                       "LABELS": [0, 1, 2]}


def test_get_dataloader():
    configuration = get_dataloader(file_name="get_dataloader_data.json", section="DATALOADER", main_dir=data_dir)
    assert configuration == GET_DATALOADER_DATA
