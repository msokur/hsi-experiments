from typing import Union

from data_utils.data_loaders.data_loader_dyn import DataLoaderDyn
from data_utils.data_loaders.data_loader_whole_dyn import DataLoaderWhole
from data_utils.smoothing import MedianFilter, GaussianFilter
from data_utils.scaler import NormalizerScaler, StandardScaler, StandardScalerTransposed
from cross_validators.cross_validator_normal import CrossValidationNormal
# from cross_validators.cross_validator_spain import CrossValidatorSpain
# from cross_validators.cross_validator_experiment import CrossValidatorExperiment
# from cross_validators.cross_validator_postprocessing import CrossValidatorPostProcessing
from trainers.trainer_tuner import TrainerTuner
from trainers.trainer_easy import TrainerEasy
from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs
from evaluation.evaluation_binary import EvaluationBinary
from evaluation.evaluation_multiclass import EvaluationMulticlass
from data_utils.data_loaders.dat_file import DatFile
from data_utils.data_loaders.mat_file import MatFile
from data_utils import border


def get_trainer(typ: str, **kwargs) -> Union[TrainerTuner, TrainerEasy, TrainerEasySeveralOutputs]:
    if typ == "Tuner":
        print('TrainerTuner')
        return TrainerTuner(**kwargs)
    elif typ == "Easy":
        print('TrainerEasy')
        return TrainerEasy(**kwargs)
    elif typ == "SeveralOutput":
        print('TrainerEasySeveralOutputs')
        return TrainerEasySeveralOutputs(**kwargs)

    value_error("Trainer", typ)


def get_data_loader(typ: str, **kwargs) -> Union[DataLoaderDyn, DataLoaderWhole]:
    if typ == "normal":
        return DataLoaderDyn(**kwargs)
    elif typ == "whole":
        return DataLoaderWhole(**kwargs)

    value_error("Data Loader", typ)


def get_whole_analog_of_data_loader(original_database: str) -> str:
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


def get_evaluation(labels: list, *args, **kwargs) -> Union[EvaluationBinary, EvaluationMulticlass]:
    print('labels', labels)
    if len(labels) == 2:
        print('Get EvaluationBinary')
        return EvaluationBinary(*args, **kwargs)
    elif len(labels) > 2:
        print('Get EvaluationMulticlass')
        return EvaluationMulticlass(*args, **kwargs)

    value_error("evaluation", "labels length < 2")


def get_smoother(typ: str, *args, **kwargs) -> Union[MedianFilter, GaussianFilter]:
    if typ == "median_filter":
        print("Smooth spectrum with median filter!")
        return MedianFilter(*args, **kwargs)
    elif typ == "gaussian_filter":
        print("Smooth spectrum with gaussian filter!")
        return GaussianFilter(*args, **kwargs)
    value_error("smoother", typ)


def get_scaler(typ: str, *args, **kwargs) -> Union[NormalizerScaler, StandardScaler, StandardScalerTransposed]:
    if typ == 'l2_norm':
        return NormalizerScaler(*args, **kwargs)
    elif typ == 'svn':
        print('StandardScaler')
        return StandardScaler(*args, **kwargs)
    elif typ == 'svn_T':
        return StandardScalerTransposed(*args, **kwargs)
    value_error("scaler", typ)


def get_pixel_detection(typ: str) -> Union[border.detect_border, border.detect_core]:
    if typ == "detect_border":
        return border.detect_border
    elif typ == "detect_core":
        return border.detect_core
    value_error("pixel detection", typ)


def get_cross_validator(typ: str, *args, **kwargs) -> CrossValidationNormal:
    if typ == "normal":
        return CrossValidationNormal()
    '''elif typ == "spain":
        return CrossValidatorSpain(*args, **kwargs)
    elif typ == "postprocessing":
        return CrossValidatorPostProcessing(*args, **kwargs)
    elif typ == "experiment":
        return CrossValidatorExperiment()'''

    value_error("Cross validator", typ)


def get_extension_loader(typ: str, **kwargs) -> Union[DatFile, MatFile]:
    if typ == ".dat":
        return DatFile(**kwargs)
    elif typ == ".mat":
        return MatFile(**kwargs)
    else:
        raise ValueError(f"For file extension {typ} is no implementation!")


def value_error(modul: str, typ: str):
    raise ValueError(f"Error! No corresponding {modul} for {typ}")


if __name__ == '__main__':
    pass
