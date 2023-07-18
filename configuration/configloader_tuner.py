import keras_tuner as kt

from models.kt_inception_model import InceptionTunerModel3D, InceptionTunerModel1D
from models.kt_paper_model import PaperTunerModel3D, PaperTunerModel1D

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


def get_tuner(tuner: str, file: str, section: str):
    if tuner in TUNER.keys():
        return TUNER[tuner]
    else:
        raise ValueError(f"In file {file}, section {section} 'TUNER' was wrongly written = "
                         f"doesn't correspond to any of 'RandomSearch', 'BayesianOptimization' or 'Hyperband'")


def get_new_dict(load_list: list, available: dict, name: str):
    optimizer_dict = {}
    if len(load_list) > 0:
        for optimizer in load_list:
            if optimizer in available.keys():
                optimizer_dict[optimizer] = available[optimizer]
            else:
                raise ValueError(f"{name} '{optimizer}' not available!")
    else:
        optimizer_dict = available.copy()

    return optimizer_dict
