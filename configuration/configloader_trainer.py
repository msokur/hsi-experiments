from typing import Tuple
import warnings

from util import tf_metric_multiclass, tf_metrics_binary
from models.inception_model import InceptionModel1D, InceptionModel3D
from models.paper_model import PaperModel1D, PaperModel3D

from configuration.configloader_base import read_config

from configuration.keys import TrainerKeys as TK

CUSTOM_OBJECTS_MULTI = {"F1_score": tf_metric_multiclass.F1_score}

CUSTOM_OBJECTS_BINARY = {"F1_score": tf_metrics_binary.F1_score}

MODELS_3D = {"paper_model": PaperModel3D().get_model,
             "inception_model": InceptionModel3D().get_model}

MODELS_1D = {"paper_model": PaperModel1D().get_model,
             "inception_model": InceptionModel1D().get_model}


def read_trainer_config(file: str, section: str, d3: bool, classes: list) -> dict:
    trainer = read_config(file=file, section=section)

    trainer[TK.CUSTOM_OBJECTS], trainer[TK.CUSTOM_OBJECTS_LOAD] = get_metric_objects(metrics=trainer[TK.CUSTOM_OBJECTS],
                                                                                     multiclass=len(classes) > 2)

    model = get_model(d3=d3, typ=trainer[TK.TYPE])

    if trainer[TK.MODEL] in model:
        trainer[TK.MODEL_CONFIG] = read_config(file=file, section=trainer[TK.MODEL_PARAMS])
        trainer[TK.MODEL] = model[trainer[TK.MODEL]]
    else:
        raise ValueError(f"Model '{trainer[TK.MODEL]}', not implemented!")

    if trainer[TK.TYPE] == "Tuner":
        from configuration.configloader_tuner import get_tuner, get_new_dict
        trainer[TK.TUNER] = get_tuner(tuner=trainer[TK.TUNER], file=file, section=section)

        trainer[TK.MODEL_CONFIG]["OPTIMIZER"] = get_new_dict(load_list=trainer[TK.MODEL_CONFIG][TK.TUNER_OPTIMIZER],
                                                             name="Optimizer")
        trainer[TK.MODEL_CONFIG]["ACTIVATION"] = get_new_dict(load_list=trainer[TK.MODEL_CONFIG][TK.TUNER_ACTIVATION],
                                                              name="Activation")

    return trainer


def get_metric_objects(metrics: dict, multiclass: bool) -> Tuple[dict, dict]:
    custom_objects = get_metric(multiclass=multiclass)

    obj_load_dict = {}
    obj_dict = {}
    for obj, args in metrics.items():
        if obj in custom_objects:
            obj_load_dict[obj] = custom_objects[obj]
            for idx, arg in enumerate(args):
                obj_dict[idx] = {"metric": custom_objects[obj], "args": arg}
        else:
            warnings.warn(f"WARNING! Custom object '{obj}', not implemented!")

    return obj_dict, obj_load_dict


def get_metric(multiclass: bool) -> dict:
    if multiclass:
        return CUSTOM_OBJECTS_MULTI
    else:
        return CUSTOM_OBJECTS_BINARY


def get_model(d3: bool, typ: str) -> dict:
    if d3:
        if typ == "Tuner":
            from configuration.configloader_tuner import TUNER_MODEL_3D
            return TUNER_MODEL_3D
        else:
            return MODELS_3D
    else:
        if typ == "Tuner":
            from configuration.configloader_tuner import TUNER_MODEL_1D
            return TUNER_MODEL_1D
        else:
            return MODELS_1D
