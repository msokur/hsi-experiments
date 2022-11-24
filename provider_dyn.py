# import config

from data_utils.data_loaders.data_loader_dyn import DataLoaderDyn
from data_utils.data_loaders.data_loader_whole_dyn import DataLoaderWhole
from data_utils.smoothing import MedianFilter, GaussianFilter
from data_utils.scaler import NormalizerScaler, StandardScaler, StandardScalerTransposed
# import trainers.trainer_easy as trainer_easy
# import trainers.trainer_easy_several_outputs as trainer_easy_several_outputs
# from evaluation.evaluation_binary import EvaluationBinary
# from evaluation.evaluation_multiclass import EvaluationMulticlass
# import models.keras_tuner_model as keras_tuner_model
# import models.keras_tuner_models_with_ones as keras_tuner_models_with_ones
from data_utils import border


'''def get_trainer(**kwargs):
    if config.RESTORE_MODEL & config.WITH_TUNING:
        raise ValueError("Custom Error! Choose if restore(config.RESTORE_MODEL) or tune(config.WITH_TUNING) "
                         "model! They could not be simultaneously True")

    if config.WITH_TUNING:
        from trainers.trainer_tuner import TrainerTuner
        print('TrainerTuner')
        return TrainerTuner(**kwargs)

    if config.DATABASE == 'bea_colon':
        print('TrainerEasy')
        return trainer_easy.TrainerEasy(**kwargs)

    if 'bea' in config.DATABASE:
        print('TrainerEasySeveralOutputs')
        return trainer_easy_several_outputs.TrainerEasySeveralOutputs(**kwargs)

    if 'hno' in config.DATABASE:
        print('TrainerEasySeveralOutputs')
        return trainer_easy_several_outputs.TrainerEasySeveralOutputs(**kwargs)

    return trainer_easy.TrainerEasy(**kwargs)'''


def get_data_loader(typ: str, loader_config: dict, path_conf: dict):
    if typ == "normal":
        return DataLoaderDyn(loader_conf=loader_config, path_conf=path_conf)
    elif typ == "whole":
        return DataLoaderWhole(loader_conf=loader_config, path_conf=path_conf)

    raise ValueError(f"Error! Dataloader type {typ} specified wrong")


def get_whole_analog_of_data_loader(original_database):
    analog = ""
    if original_database == 'colon':
        analog = 'colon_whole'
    if original_database == 'bea_brain':
        analog = 'bea_brain_whole'
    if original_database == 'bea_eso':
        analog = 'bea_eso_whole'
    if original_database == 'bea_colon':
        analog = 'bea_colon_whole'
    if original_database == 'hno':
        analog = 'hno_whole'

    if analog == "":
        raise ValueError(f"We didn't found an analog whole database for {original_database}")

    print(f"For {original_database} we have found {analog} whole database")

    return analog


'''def get_evaluation(*args, **kwargs):
    labels = config.LABELS_OF_CLASSES_TO_TRAIN
    print('labels', labels)
    if len(labels) == 2:
        print('Get EvaluationBinary')
        return EvaluationBinary(*args, **kwargs)
    print('Get EvaluationMulticlass')
    return EvaluationMulticlass(*args, **kwargs)'''


'''def get_keras_tuner_model():
    if config.TUNER_MODEL == 'KerasTunerModel':
        return keras_tuner_model.KerasTunerModel()
    if config.TUNER_MODEL == 'KerasTunerModelOnes':
        return keras_tuner_models_with_ones.KerasTunerModelOnes()

    raise ValueError(f'Error! Tuner model type {config.TUNER_MODEL} specified wrong (either in config.py or in '
                     f'provider.py)')'''


def get_smoother(typ: str, *args, **kwargs):
    if typ == 'median_filter':
        return MedianFilter(*args, **kwargs)
    elif typ == 'gaussian_filter':
        return GaussianFilter(*args, **kwargs)
    value_error("smoother", typ)


def get_scaler(typ: str, *args, **kwargs):
    if typ == 'l2_norm':
        return NormalizerScaler(*args, **kwargs)
    elif typ == 'svn':
        print('StandardScaler')
        return StandardScaler(*args, **kwargs)
    elif typ == 'svn_T':
        return StandardScalerTransposed(*args, **kwargs)
    value_error("scaler", typ)


def get_pixel_detection(typ: str):
    if typ == "detect_border":
        return border.detect_border
    elif typ == "detect_core":
        return border.detect_core
    value_error("pixel detection", typ)


def value_error(modul: str, typ: str):
    raise ValueError(f"Error! No corresponding {modul} for {typ}")


if __name__ == '__main__':
    pass