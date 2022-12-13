import json
import platform
import os

import keras_tuner as kt
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from util import tf_metric_multiclass, tf_metrics
from models.inception_model import inception1d_model, inception3d_model
from models.paper_model import paper1d_model, paper3d_model
from models.kt_model import InceptionTunerModel3D, InceptionTunerModel1D

CONCAT_KEY = "CONCAT_WITH_"
CONVERT_KEY = ["MASK_COLOR", "TISSUE_LABELS", "PLOT_COLORS"]
CUSTOM_OBJECTS_MULTI = {
    "F1_score": tf_metric_multiclass.F1_score
}
CUSTOM_OBJECTS_BINARY = {
    "F1_score": tf_metrics.f1_m
}
MODELS_3D = {
    "paper_model": paper3d_model,
    "inception_model": inception3d_model
}
MODELS_1D = {
    "paper_model": paper1d_model,
    "inception_model": inception1d_model
}
TUNER_MODEL_3D = {
    "inception_model": InceptionTunerModel3D
}
TUNER_MODEL_1D = {
    "inception_model": InceptionTunerModel1D
}
TUNER = {
    "RandomSearch": kt.RandomSearch,
    "BayesianOptimization": kt.BayesianOptimization,
    "Hyperband": kt.Hyperband
}
OPTIMIZER = {"adadelta": Adadelta,
             "adagrad": Adagrad,
             "adam": Adam,
             "adamax": Adamax,
             "ftrl": Ftrl,
             "nadam": Nadam,
             "rms": RMSprop,
             "sgd": SGD
             }


def read_path_config(file: str, system_mode: str, database: str) -> dict:
    data_system = read_config(file=file, section=system_mode)
    data_database = read_config(file=file, section=database)
    data_base = read_config(file=file, section="BASE_PATH")

    path_dict = concat_dict(data_system, data_base)
    path_dict = get_database_paths(path_dict=path_dict, to_prefix=data_database)
    if platform.system() == 'Windows':
        path_dict["SYSTEM_PATHS_DELIMITER"] = "\\"
    else:
        path_dict["SYSTEM_PATHS_DELIMITER"] = "/"

    return path_dict


def get_database_paths(path_dict: dict, to_prefix: dict) -> dict:
    path_dict_ = path_dict.copy()
    for key, values in to_prefix.items():
        if CONCAT_KEY in key:
            prefix = key.split(CONCAT_KEY)[1]
            if prefix in path_dict_.keys():
                temp = set_prefix(prefix=path_dict_[prefix], database=values)
                path_dict_ = concat_dict(path_dict_, temp)
            else:
                raise ValueError(f"Check the order form your Database paths, prefix: {prefix} not found!")
        else:
            path_dict_[key] = os.path.join(*values)

    return path_dict_


def set_prefix(prefix: str, database: dict) -> dict:
    prefixed_paths = {}
    for key, value in database.items():
        prefixed_paths[key] = os.path.join(prefix, *value)

    return prefixed_paths


def concat_dict(dict1: dict, dict2: dict) -> dict:
    dict_temp = dict1.copy()
    for key, value in dict2.items():
        if key not in dict1:
            dict_temp[key] = value
        else:
            raise ValueError(f'The key {key} is already in the Dictionary!')
    return dict_temp


def convert_key_to_int(str_dict: dict):
    int_dict = {}
    for key, value in str_dict.items():
        int_dict[int(key)] = value
    return int_dict


def read_config(file: str, section: str) -> dict:
    with open(file, "r") as config_file:
        data = json.load(config_file)

    if section in data:
        data_ = data[section]
        for convert in CONVERT_KEY:
            if convert in data_:
                data_[convert] = convert_key_to_int(data_[convert])
        return data_
    else:
        raise ValueError(f'Section {section}, not found in the {file} file!')


def read_trainer_config(file: str, section: str, d3: bool, classes: list) -> dict:
    trainer = read_config(file=file, section=section)
    if len(classes) > 2:
        custom_objects = CUSTOM_OBJECTS_MULTI
    else:
        custom_objects = CUSTOM_OBJECTS_BINARY

    obj_load_dict = {}
    obj_dict = {}
    for obj, args in trainer["CUSTOM_OBJECTS"].items():
        if obj in custom_objects:
            obj_load_dict[obj] = custom_objects[obj]
            for idx, arg in enumerate(args):
                obj_dict[idx] = {"metric": custom_objects[obj], "args": arg}
        else:
            print(f"WARNING! Custom object {obj}, not implemented!")
    trainer["CUSTOM_OBJECTS"] = obj_dict
    trainer["CUSTOM_OBJECTS_LOAD"] = obj_load_dict

    if d3:
        if trainer["TYPE"] == "Tuner":
            models = TUNER_MODEL_3D
        else:
            models = MODELS_3D
    else:
        if trainer["TYPE"] == "Tuner":
            models = TUNER_MODEL_1D
        else:
            models = MODELS_1D

    if trainer["MODEL"] in models:
        trainer["MODEL_CONFIG"] = read_config(file=file, section=trainer["MODEL_PARAMS"])
        trainer["MODEL"] = models[trainer["MODEL"]]
    else:
        raise ValueError(f"Model {trainer['MODEL']}, not implemented!")

    if trainer["TYPE"] == "Tuner":
        if trainer["TUNER"] in TUNER.keys():
            trainer["TUNER"] = TUNER[trainer["TUNER"]]
        else:
            raise ValueError(f"In file {file}, section {section} 'TUNER' was wrongly written = "
                             f"doesn't correspond to any of 'RandomSearch', 'BayesianOptimization' or 'Hyperband'")
        optimizer_dict = {}
        if len(trainer["MODEL_CONFIG"]["OPTIMIZER"]) > 0:
            for optimizer in trainer["MODEL_CONFIG"]["OPTIMIZER"]:
                if optimizer in OPTIMIZER.keys():
                    optimizer_dict[optimizer] = OPTIMIZER[optimizer]
                else:
                    raise ValueError(f"Optimizer '{optimizer}' not available!")
        else:
            optimizer_dict = OPTIMIZER.copy()

        trainer["MODEL_CONFIG"]["OPTIMIZER"] = optimizer_dict

    return trainer


def read_cv_config(file: str, section: str) -> dict:
    cv_base = read_config(file=file, section="BASE")
    cv_section = read_config(file=file, section=section)
    cv = concat_dict(dict1=cv_base, dict2=cv_section)
    return cv
