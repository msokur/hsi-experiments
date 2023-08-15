# modules are imported inside functions to avoid circular imports


def get_trainer(typ: str, **kwargs):
    from trainers.trainer_tuner import TrainerTuner
    from trainers.trainer_easy import TrainerEasy
    from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs

    if typ == "Tuner":
        print("TrainerTuner")
        return TrainerTuner(**kwargs)
    elif typ == "Easy":
        print("TrainerEasy")
        return TrainerEasy(**kwargs)
    elif typ == "SeveralOutput":
        print("TrainerEasySeveralOutputs")
        return TrainerEasySeveralOutputs(**kwargs)

    value_error("Trainer", typ)


def get_data_loader(typ: str, **kwargs):
    from data_utils.data_loaders.data_loader_npz import DataLoaderNPZ
    from data_utils.data_loaders.data_loader_whole import DataLoaderWhole
    
    if typ == "normal":
        return DataLoaderNPZ(**kwargs)
    elif typ == "whole":
        return DataLoaderWhole(**kwargs)

    value_error("Data Loader", typ)


def get_whole_analog_of_data_loader(original_database):    # maybe out of date
    analog = ""
    if original_database == 'normal':
        analog = 'whole'
    if original_database == 'colon':
        analog = 'colon_whole'
    if original_database == 'bea_brain':
        analog = 'bea_brain_whole'
    if original_database == 'bea_eso':
        analog = 'bea_eso_whole'
    if original_database == 'bea_colon':
        analog = 'bea_colon_whole'
    if original_database == 'hno':
        analog = "hno_whole"

    if analog == "":
        raise ValueError(f"We didn't found an analog whole database for {original_database}")

    print(f"For '{original_database}' we have found '{analog}' whole database")

    return analog


def get_evaluation(labels: list, *args, **kwargs):
    from evaluation.evaluation_binary import EvaluationBinary
    from evaluation.evaluation_multiclass import EvaluationMulticlass
    
    if len(labels) == 2:
        print('Get EvaluationBinary')
        return EvaluationBinary(*args, **kwargs)
    elif len(labels) > 2:
        print('Get EvaluationMulticlass')
        return EvaluationMulticlass(*args, **kwargs)

    value_error("evaluation", "labels length < 2")


def get_smoother(typ: str, *args, **kwargs):
    from data_utils.smoothing import MedianFilter, GaussianFilter
    
    if typ == "median_filter":
        print("Smooth spectrum with median filter!")
        return MedianFilter(*args, **kwargs)
    elif typ == "gaussian_filter":
        print("Smooth spectrum with gaussian filter!")
        return GaussianFilter(*args, **kwargs)
    
    value_error("smoother", typ)


def get_scaler(typ: str, *args, **kwargs):
    from data_utils.scaler import NormalizerScaler, StandardScaler, StandardScalerTransposed

    if typ == 'l2_norm':
        return NormalizerScaler(*args, **kwargs)
    elif typ == 'svn':
        return StandardScaler(*args, **kwargs)
    elif typ == 'svn_T':
        return StandardScalerTransposed(*args, **kwargs)
    
    value_error("scaler", typ)


def get_pixel_detection(typ: str):
    from data_utils import border
    
    if typ == "detect_border":
        return border.detect_border
    elif typ == "detect_core":
        return border.detect_core
    value_error("pixel detection", typ)


def get_cross_validator(typ: str, *args, **kwargs):
    from cross_validators.cross_validator_normal import CrossValidationNormal
    from cross_validators.cross_validator_postprocessing import CrossValidatorPostProcessing
    
    if typ == "normal":
        return CrossValidationNormal()
    elif typ == "spain":
        from cross_validators.cross_validator_spain import CrossValidatorSpain
        return CrossValidatorSpain(*args, **kwargs)
    elif typ == "postprocessing":
        return CrossValidatorPostProcessing(*args, **kwargs)
    elif typ == "experiment":
        from cross_validators.cross_validator_experiment import CrossValidatorExperiment
        return CrossValidatorExperiment()

    value_error("Cross validator", typ)


def get_extension_loader(typ: str, **kwargs):
    from data_utils.data_loaders.dat_file import DatFile
    from data_utils.data_loaders.mat_file import MatFile

    if typ == ".dat":
        return DatFile(**kwargs)
    elif typ == ".mat":
        return MatFile(**kwargs)
    else:
        value_error(modul="file extension", typ=typ)


def get_data_archive(typ: str, archive_path: str, archive_name: str = None, chunks: tuple = None):
    from data_utils.data_archive import DataArchiveNPZ, DataArchiveZARR

    if typ == "npz":
        return DataArchiveNPZ(archive_path=archive_path)
    elif typ == "zarr":
        return DataArchiveZARR(archive_path=archive_path, archive_name=archive_name, chunks=chunks)
    else:
        value_error(modul="data archive", typ=typ)


def value_error(modul: str, typ: str):
    raise ValueError(f"Error! No corresponding {modul} for {typ}")


if __name__ == '__main__':
    pass
