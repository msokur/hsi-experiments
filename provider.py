# modules are imported inside functions to avoid circular imports


def get_trainer(typ: str, **kwargs):
    from trainers.trainer_easy import TrainerEasy
    from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs

    if typ == "Tuner":
        print('TrainerTuner')
        from trainers.trainer_tuner import TrainerTuner
        return TrainerTuner(**kwargs)
    elif typ == "Easy":
        print('TrainerEasy')
        return TrainerEasy(**kwargs)
    elif typ == "SeveralOutput":
        print('TrainerEasySeveralOutputs')
        return TrainerEasySeveralOutputs(**kwargs)

    value_error("Trainer", typ)


def get_data_loader(typ: str, **kwargs):
    from data_utils.data_loaders.data_loader import DataLoader
    from data_utils.data_loaders.data_loader_whole import DataLoaderWhole
    #from data_utils.data_loaders.archive.data_loader_colon import DataLoaderColon
    #from data_utils.data_loaders.archive.data_loader_mat import DataLoaderMat
    #from data_utils.data_loaders.archive.data_loader_mat_brain import DataLoaderMatBrain
    #from data_utils.data_loaders.archive.data_loader_mat_colon import DataLoaderMatColon
    #from data_utils.data_loaders.archive.data_loader_whole_colon import DataLoaderWholeColon
    #from data_utils.data_loaders.archive.data_loader_whole_mat import DataLoaderWholeMat
    #from data_utils.data_loaders.archive.data_loader_hno import DataLoaderHNO
    #from data_utils.data_loaders.archive.data_loader_whole_hno import DataLoaderWholeHNO
    
    if typ == "normal":
        return DataLoader(**kwargs)
    elif typ == "whole":
        return DataLoaderWhole(**kwargs)
    
    '''if typ == 'colon':
        print('DataLoaderColon')
        return DataLoaderColon(**kwargs)
    if typ == 'bea_eso':
        print('DataLoaderMat')
        return DataLoaderMat(**kwargs)
    if typ == 'bea_brain':
        print('DataLoaderMatBrain')
        return DataLoaderMatBrain(**kwargs)
    if typ == 'bea_colon':
        print('DataLoaderMatColon')
        return DataLoaderMatColon(**kwargs)
    if typ == 'hno':
        print('DataLoaderHNO')
        return DataLoaderHNO(**kwargs)

    # analogs for whole cubes
    if typ == 'colon_whole':
        print('DataLoaderWholeColon')
        return DataLoaderWholeColon(**kwargs)
    if typ == 'hno_whole':
        print('DataLoaderWhole with HNO')
        return DataLoaderWholeHNO(**kwargs)
    if typ == 'bea_brain_whole':
        print('DataLoaderWholeMat with Brain')
        return DataLoaderWholeMat(DataLoaderMatBrain(**kwargs), **kwargs)
    if typ == 'bea_eso_whole':
        print('DataLoaderWholeMat with Eso')
        return DataLoaderWholeMat(DataLoaderMat(**kwargs), **kwargs)
    if typ == 'bea_colon_whole':
        print('DataLoaderWholeMat with Colon')
        return DataLoaderWholeMat(DataLoaderMatColon(**kwargs), **kwargs)'''

    value_error("Data Loader", typ)


def get_whole_analog_of_data_loader(original_database):    # maybe out of date
    analog = ""
    if original_database == 'normal':
        analog = 'whole'
    '''if original_database == 'colon':
        analog = 'colon_whole'
    if original_database == 'bea_brain':
        analog = 'bea_brain_whole'
    if original_database == 'bea_eso':
        analog = 'bea_eso_whole'
    if original_database == 'bea_colon':
        analog = 'bea_colon_whole'
    if original_database == 'hno':
        analog = "hno_whole"'''

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
    
    print('Get EvaluationMulticlass')
    return EvaluationMulticlass(*args, **kwargs)


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
    from data_utils.scaler import NormalizerScaler, StandardScalerTransposed

    if typ == 'l2_norm':
        return NormalizerScaler(*args, **kwargs)
    elif typ == 'svn':
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
        return CrossValidationNormal(*args, **kwargs)
    elif typ == "spain":
        from cross_validators.cross_validator_spain import CrossValidatorSpain
        return CrossValidatorSpain(*args, **kwargs)
    elif typ == "postprocessing":
        return CrossValidatorPostProcessing(*args, **kwargs)
    elif typ == "experiment":
        from cross_validators.cross_validator_experiment import CrossValidatorExperiment
        return CrossValidatorExperiment(*args, **kwargs)

    value_error("Cross validator", typ)


def get_extension_loader(typ: str, **kwargs):
    from data_utils.data_loaders.dat_file import DatFile
    from data_utils.data_loaders.mat_file import MatFile

    if typ == ".dat":
        return DatFile(**kwargs)
    elif typ == ".mat":
        return MatFile(**kwargs)
    else:
        raise ValueError(f"For file extension {typ} is no implementation!")
        
def get_keras_tuner_model(typ: str):  #maybe out of date
    if typ == 'KerasTunerModel':
        return keras_tuner_model.KerasTunerModel()
    if typ == 'KerasTunerModelOnes':
        return keras_tuner_models_with_ones.KerasTunerModelOnes()

    raise ValueError(f'Error! Tuner model type {typ} specified wrong (either in config.py or in '
                     f'provider.py)')


def value_error(modul: str, typ: str):
    raise ValueError(f"Error! No corresponding {modul} for {typ}")


if __name__ == '__main__':
    pass
