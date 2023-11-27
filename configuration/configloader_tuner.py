import warnings

import keras_tuner as kt
from models.kt_inception_model import InceptionTunerModel3D, InceptionTunerModel1D
from models.kt_paper_model import PaperTunerModel3D, PaperTunerModel1D

from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from tensorflow.keras.activations import relu, tanh, selu, exponential, elu

# --- Tuner model for patch sized data
TUNER_MODEL_3D = {"paper_model": PaperTunerModel3D,
                  "inception_model": InceptionTunerModel3D}

# --- Tuner model for data without patch size
TUNER_MODEL_1D = {"paper_model": PaperTunerModel1D,
                  "inception_model": InceptionTunerModel1D}

# --- keras tuner
TUNER = {"RandomSearch": kt.RandomSearch,
         "BayesianOptimization": kt.BayesianOptimization,
         "Hyperband": kt.Hyperband}

# --- Optimizer and Activation
LOADS = {"Optimizer": {"adadelta": Adadelta,
                       "adagrad": Adagrad,
                       "adam": Adam,
                       "adamax": Adamax,
                       "ftrl": Ftrl,
                       "nadam": Nadam,
                       "rms": RMSprop,
                       "sgd": SGD},
         "Activation": {"relu": relu,
                        "tanh": tanh,
                        "selu": selu,
                        "exponential": exponential,
                        "elu": elu}}


def get_tuner(tuner: str, file: str, section: str):
    """Get a keras tuner

    :param tuner: Name of tuner to load
    :param file: Json file name
    :param section: Section name

    :return: A keras tuner

    :raises ValueError: When name in 'tuber' not match with the implemented tuner
    """
    if tuner in TUNER.keys():
        return TUNER[tuner]
    else:
        raise ValueError(f"In file '{file}', section '{section}' 'TUNER' was wrongly written = "
                         f"doesn't correspond to any of 'RandomSearch', 'BayesianOptimization' or 'Hyperband'")


def get_new_dict(load_list: list, name: str):
    """Load Optimizer or Activation functions.

    :param load_list: Names from Optimizer or Activation functions
    :param name: Optimizer or Activation

    :return: Dictionary with Optimizer or Activation functions
    """
    optimizer_dict = {}
    if len(load_list) > 0:
        for key in load_list:
            if key in LOADS[name].keys():
                optimizer_dict[key] = LOADS[name][key]
            else:
                raise ValueError(f"Key: '{key}' in {name} not available!")
    else:
        warnings.warn(f"For '{name}' no values given. Return all possible values!")
        optimizer_dict = LOADS[name].copy()

    return optimizer_dict
