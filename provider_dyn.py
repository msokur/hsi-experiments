from data_utils.data_loaders.data_loader_dyn import DataLoaderDyn
from data_utils.data_loaders.data_loader_whole_dyn import DataLoaderWhole
from data_utils.smoothing import MedianFilter, GaussianFilter
from data_utils.scaler import NormalizerScaler, StandardScaler, StandardScalerTransposed
from cross_validators.cross_validator_normal import CrossValidationNormal
from cross_validators.cross_validator_spain import CrossValidatorSpain
from cross_validators.cross_validator_experiment import CrossValidatorExperiment
from cross_validators.cross_validator_postprocessing import CrossValidatorPostProcessing
from trainers.trainer_tuner import TrainerTuner
from trainers.trainer_easy import TrainerEasy
from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs
# from evaluation.evaluation_binary import EvaluationBinary
# from evaluation.evaluation_multiclass import EvaluationMulticlass
# import models.keras_tuner_model as keras_tuner_model
# import models.keras_tuner_models_with_ones as keras_tuner_models_with_ones
from data_utils import border


def get_trainer(typ: str, **kwargs):
    if typ == "Tuner":
        print('TrainerTuner')
        return TrainerTuner(**kwargs)
    elif type == "Easy":
        print('TrainerEasy')
        return TrainerEasy(**kwargs)
    elif typ == "Several Output":
        print('TrainerEasySeveralOutputs')
        return TrainerEasySeveralOutputs(**kwargs)

    return TrainerEasy(**kwargs)


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


def get_cross_validator(typ: str, *args, **kwargs):
    if typ == "normal":
        return CrossValidationNormal(*args, **kwargs)
    elif typ == "spain":
        return CrossValidatorSpain(*args, **kwargs)
    elif typ == "postprocessing":
        return CrossValidatorPostProcessing(*args, **kwargs)
    elif typ == "experiment":
        return CrossValidatorExperiment()

    value_error("Cross validator", typ)


def value_error(modul: str, typ: str):
    raise ValueError(f"Error! No corresponding {modul} for {typ}")


if __name__ == '__main__':
    pass
