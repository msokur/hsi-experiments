import keras_tuner as kt
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

from util import tf_metric_multiclass, tf_metrics_binary
from models.inception_model import inception1d_model, inception3d_model
from models.paper_model import paper1d_model, paper3d_model
from models.kt_inception_model import InceptionTunerModel3D, InceptionTunerModel1D
from models.kt_paper_model import PaperTunerModel3D, PaperTunerModel1D
from configuration.configloader_base import read_config


CUSTOM_OBJECTS_MULTI = {
    "F1_score": tf_metric_multiclass.F1_score
}
CUSTOM_OBJECTS_BINARY = {
    "F1_score": tf_metrics_binary.F1_score
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
OPTIMIZER = {"adadelta": Adadelta,
             "adagrad": Adagrad,
             "adam": Adam,
             "adamax": Adamax,
             "ftrl": Ftrl,
             "nadam": Nadam,
             "rms": RMSprop,
             "sgd": SGD
             }


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

    model = get_trainer(d3=d3, typ=trainer["TYPE"])

    if trainer["MODEL"] in model:
        trainer["MODEL_CONFIG"] = read_config(file=file, section=trainer["MODEL_PARAMS"])
        trainer["MODEL"] = model[trainer["MODEL"]]
    else:
        raise ValueError(f"Model {trainer['MODEL']}, not implemented!")

    if trainer["TYPE"] == "Tuner":
        trainer["TUNER"] = get_tuner(tuner=trainer["TUNER"], file=file, section=section)

        trainer["MODEL_CONFIG"]["OPTIMIZER"] = get_optimizer(optimizer_list=trainer["MODEL_CONFIG"]["OPTIMIZER"])

    return trainer


def get_trainer(d3: bool, typ: str):
    if d3:
        if typ == "Tuner":
            return TUNER_MODEL_3D
        else:
            return MODELS_3D
    else:
        if typ == "Tuner":
            return TUNER_MODEL_1D
        else:
            return MODELS_1D


def get_tuner(tuner: str, file: str, section: str):
    if tuner in TUNER.keys():
        return TUNER[tuner]
    else:
        raise ValueError(f"In file {file}, section {section} 'TUNER' was wrongly written = "
                         f"doesn't correspond to any of 'RandomSearch', 'BayesianOptimization' or 'Hyperband'")


def get_optimizer(optimizer_list: list):
    optimizer_dict = {}
    if len(optimizer_list) > 0:
        for optimizer in optimizer_list:
            if optimizer in OPTIMIZER.keys():
                optimizer_dict[optimizer] = OPTIMIZER[optimizer]
            else:
                raise ValueError(f"Optimizer '{optimizer}' not available!")
    else:
        optimizer_dict = OPTIMIZER.copy()

    return optimizer_dict
