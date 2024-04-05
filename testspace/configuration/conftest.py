import os

import pytest

import keras_tuner as kt
from models.kt_inception_model import InceptionTunerModel3D, InceptionTunerModel1D
from models.kt_paper_model import PaperTunerModel3D, PaperTunerModel1D
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from tensorflow.keras.activations import relu, tanh, selu, exponential, elu

from models.paper_model import PaperModel1D, PaperModel3D
from models.inception_model import InceptionModel1D, InceptionModel3D
from util import tf_metric_multiclass, tf_metrics_binary


# --- pytest fixtures for the paths to load test data --
@pytest.fixture
def config_data_dir(main_dir: str) -> str:
    return os.path.join(main_dir, "_test_configs")


@pytest.fixture
def base_config_path(config_data_dir: str) -> str:
    return os.path.join(config_data_dir, "preprocessor_data.json")


@pytest.fixture
def cv_data_dir(config_data_dir: str) -> str:
    return os.path.join(config_data_dir, "cv_data.json")


@pytest.fixture
def dataloader_data_dir(config_data_dir: str) -> str:
    return os.path.join(config_data_dir, "dataloader_data.json")


@pytest.fixture
def paths_data_dir(config_data_dir: str) -> str:
    return os.path.join(config_data_dir, "paths_data.json")


@pytest.fixture
def trainer_data_dir(config_data_dir: str) -> str:
    return os.path.join(config_data_dir, "trainer_data.json")


# --- shared results ---
# result for base config data
@pytest.fixture
def base_config_result() -> dict:
    return {"EXECUTION_FLAGS": {"LOAD_DATA_WITH_DATALOADER": True,
                                "ADD_SAMPLE_WEIGHTS": True,
                                "SCALE": True,
                                "SHUFFLE": True},
            "DICT_NAMES": ["X", "y", "PatientName", "PatientIndex", "indexes_in_datacube", "weights"],
            "PILES_NUMBER": 100,
            "WEIGHT_FILENAME": "weights.weights",
            "FILES_TO_COPY": ["test_file.py"],
            "NORMALIZATION_TYPE": "svn",
            "SCALER_FILE": "scaler.scaler",
            "SCALER_PATH": "scaler_path"}


# result for dataloader config data
@pytest.fixture
def dataloader_result() -> dict:
    return {"TYPE": "normal",
            "FILE_EXTENSION": ".dat",
            "3D": True,
            "3D_SIZE": [3, 3],
            "FIRST_NM": 8,
            "LAST_NM": 100,
            "WAVE_AREA": 100,
            "LABELS_TO_TRAIN": [0, 1],
            "NAME_SPLIT": "_SpecCube",
            "MASK_DIFF": ["dat.dat", "mask.png"],
            "LABELS_FILENAME": "labels.labels",
            "CONTAMINATION_FILENAME": "contamination.csv",
            "SMOOTHING_TYPE": "median_filter",
            "SMOOTHING_VALUE": 5,
            "BORDER_CONFIG": {"enable": False,
                              "methode": "detect_core",
                              "depth": 5,
                              "axis": [],
                              "not_used_labels": []},
            "SPLIT_PATHS_BY": "Files",
            "CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST": 1,
            "WITH_BACKGROUND_EXTRACTION": False,
            "MASK_COLOR": {0: [[255, 255, 0]], 1: [[0, 0, 255]], 2: [[255, 0, 0]]},
            "TISSUE_LABELS": {0: "Class0", 1: "Class1", 2: "Class2"},
            "PLOT_COLORS": {0: "yellow", 1: "blue", 2: "red"},
            "LABELS": [0, 1, 2]}


# result for cross validation config data
@pytest.fixture
def cv_result() -> dict:
    return {"EXECUTION_FLAGS": {"cross_validation": True,
                                "evaluation": True},
            "FIRST_SPLIT": 0,
            "HOW_MANY_VALID_EXCLUDE": 1,
            "CHOOSE_EXCLUDED_VALID": "by_class",
            "USE_ALL_LABELS": False,
            "SAVE_PREDICTION": True,
            "SAVE_CURVES": False,
            "MODE": "RUN",
            "TYPE": "normal",
            "NAME": "CV_NAME",
            "DATABASE_ABBREVIATION": "DATABASE_NAME",
            "RESTORE_VALID_PATIENTS_FOLDER": "/home/sc.uni-leipzig.de/xyz/hsi-experiments-BA/logs/Experiment",
            "GET_CHECKPOINT_FROM_VALID": True}


# results for path config data
@pytest.fixture
def path_prefix() -> str:
    return "/work/users/xyz"


@pytest.fixture
def path_result(sys_slash: str, path_prefix: str):
    return {"CHECKPOINT_PATH": "checkpoints",
            "RESULTS_FOLDER": "results",
            "MODE": "WITH_GPU",
            "PREFIX": path_prefix,
            "MODEL_NAME_PATHS": ["/home/sc.uni-leipzig.de/xyz/hsi-experiments-BA/logs"],
            "RAW_NPZ_PATH": f"{path_prefix}{sys_slash}data_3d{sys_slash}gastric{sys_slash}3x3",
            "RAW_SOURCE_PATH": f"{path_prefix}{sys_slash}Gastric",
            "TEST_NPZ_PATH": f"{path_prefix}{sys_slash}data_3d{sys_slash}hno{sys_slash}3x3",
            "SHUFFLED_PATH": f"{path_prefix}{sys_slash}data_3d{sys_slash}gastric{sys_slash}3x3{sys_slash}shuffled",
            "BATCHED_PATH": f"{path_prefix}{sys_slash}data_3d{sys_slash}gastric{sys_slash}3x3{sys_slash}batch_sized",
            "MASK_PATH": f"{path_prefix}{sys_slash}Gastric{sys_slash}annotation",
            "SYSTEM_PATHS_DELIMITER": sys_slash}


# results tuner data
@pytest.fixture
def activation() -> dict:
    return {"relu": relu,
            "tanh": tanh,
            "selu": selu,
            "exponential": exponential,
            "elu": elu}


@pytest.fixture
def optimizer() -> dict:
    return {"adadelta": Adadelta,
            "adagrad": Adagrad,
            "adam": Adam,
            "adamax": Adamax,
            "ftrl": Ftrl,
            "nadam": Nadam,
            "rms": RMSprop,
            "sgd": SGD}


@pytest.fixture
def tuner() -> dict:
    return {"RandomSearch": kt.RandomSearch,
            "BayesianOptimization": kt.BayesianOptimization,
            "Hyperband": kt.Hyperband}


@pytest.fixture
def model_tuner() -> dict:
    return {"1D": {"paper_model": PaperTunerModel1D,
                   "inception_model": InceptionTunerModel1D},
            "3D": {"paper_model": PaperTunerModel3D,
                   "inception_model": InceptionTunerModel3D}}


# result trainer data
@pytest.fixture
def model_normal() -> dict:
    return {"1D": {"paper_model": PaperModel1D,
                   "inception_model": InceptionModel1D},
            "3D": {"paper_model": PaperModel3D,
                   "inception_model": InceptionModel3D}}


@pytest.fixture
def metric() -> dict:
    return {"binary": {"F1_score": tf_metrics_binary.F1_score},
            "multi": {"F1_score": tf_metric_multiclass.F1_score}}


@pytest.fixture
def trainer_normal_base() -> dict:
    return {"TYPE": "SeveralOutput",
            "RESTORE": False,
            "FILES_TO_COPY": [["*.py"]],
            "WITH_SAMPLE_WEIGHTS": False,
            "MODEL": "",
            "MODEL_CONFIG": {"DROPOUT": 0.1},
            "MODEL_PARAMS": "model_param_2",
            "LEARNING_RATE": 1e-4,
            "CUSTOM_OBJECTS": {0: {"metric": "",
                                   "args": {"name": "f1_score_weighted", "average": "weighted"}}},
            "CUSTOM_OBJECTS_LOAD": {},
            "BATCH_SIZE": 500,
            "SPLIT_FACTOR": 0.8,
            "EPOCHS": 50,
            "SMALLER_DATASET": False,
            "MODEL_CHECKPOINT": {"monitor": "val_f1_score_weighted",
                                 "save_best_only": True,
                                 "mode": "max"},
            "EARLY_STOPPING": {"enable": True,
                               "monitor": "val_f1_score_weighted",
                               "mode": "max",
                               "min_delta": 0,
                               "patience": 5,
                               "restore_best_weights": True}}


@pytest.fixture
def trainer_tuner_base(trainer_normal_base: dict, activation: dict, optimizer: dict, tuner: dict) -> dict:
    base_tuner = trainer_normal_base.copy()
    base_tuner["TYPE"] = "Tuner"
    base_tuner["MODEL_CONFIG"] = {"DROPOUT": 0.1, "OPTIMIZER": optimizer, "ACTIVATION": activation}
    base_tuner["MODEL_PARAMS"] = "model_param_1"
    base_tuner.update({"TUNER": tuner["BayesianOptimization"],
                       "TUNER_PARAMS": {
                           "objective": {"name": "f1_score_weighted", "direction": "max"},
                           "max_trials": 20,
                           "overwrite": True
                       },
                       "TUNER_EPOCHS": 10})
    return base_tuner
