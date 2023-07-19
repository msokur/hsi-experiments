from util import tf_metric_multiclass, tf_metrics_binary
from models.inception_model import InceptionModel1D, InceptionModel3D
from models.paper_model import PaperModel1D, PaperModel3D

from configuration.configloader_base import read_config


CUSTOM_OBJECTS_MULTI = {
    "F1_score": tf_metric_multiclass.F1_score
}
CUSTOM_OBJECTS_BINARY = {
    "F1_score": tf_metrics_binary.F1_score
}
MODELS_3D = {
    "paper_model": PaperModel3D().get_model,
    "inception_model": InceptionModel3D().get_model
}
MODELS_1D = {
    "paper_model": PaperModel1D().get_model,
    "inception_model": InceptionModel1D().get_model
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
        from configuration.configloader_tuner import get_tuner, get_new_dict
        trainer["TUNER"] = get_tuner(tuner=trainer["TUNER"], file=file, section=section)

        trainer["MODEL_CONFIG"]["OPTIMIZER"] = get_new_dict(load_list=trainer["MODEL_CONFIG"]["OPTIMIZER"],
                                                            name="Optimizer")
        trainer["MODEL_CONFIG"]["ACTIVATION"] = get_new_dict(load_list=trainer["MODEL_CONFIG"]["ACTIVATION"],
                                                             name="Activation")

    return trainer


def get_trainer(d3: bool, typ: str):
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
