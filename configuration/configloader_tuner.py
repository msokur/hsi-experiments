import keras_tuner as kt

from models.kt_inception_model import InceptionTunerModel3D, InceptionTunerModel1D
from models.kt_paper_model import PaperTunerModel3D, PaperTunerModel1D

from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from tensorflow.keras.activations import relu, tanh, selu, exponential, elu

TUNER_MODEL_3D = {
    "paper_model": PaperTunerModel3D,
    "inception_model": InceptionTunerModel3D
}
TUNER_MODEL_1D = {
    "paper_model": PaperTunerModel1D,
    "inception_model": InceptionTunerModel1D
}
TUNER = {
    "RandomSearch": kt.RandomSearch,
    "BayesianOptimization": kt.BayesianOptimization,
    "Hyperband": kt.Hyperband
}

LOADS = {"Optimizer": {"adadelta": Adadelta,
                       "adagrad": Adagrad,
                       "adam": Adam,
                       "adamax": Adamax,
                       "ftrl": Ftrl,
                       "nadam": Nadam,
                       "rms": RMSprop,
                       "sgd": SGD
                       },
         "Activation": {"relu": relu,
                        "tanh": tanh,
                        "selu": selu,
                        "exponential": exponential,
                        "elu": elu}}


def get_tuner(tuner: str, file: str, section: str):
    if tuner in TUNER.keys():
        return TUNER[tuner]
    else:
        raise ValueError(f"In file {file}, section {section} 'TUNER' was wrongly written = "
                         f"doesn't correspond to any of 'RandomSearch', 'BayesianOptimization' or 'Hyperband'")


def get_new_dict(load_list: list, name: str):
    optimizer_dict = {}
    if len(load_list) > 0:
        for key in load_list:
            if key in LOADS[name].keys():
                optimizer_dict[key] = LOADS[name][key]
            else:
                raise ValueError(f"{name} '{key}' not available!")
    else:
        optimizer_dict = LOADS[name].copy()

    return optimizer_dict
