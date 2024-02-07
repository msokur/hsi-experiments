# modules are imported inside functions to avoid circular imports


def get_trainer(typ: str, config, *args, **kwargs):
    from trainers.trainer_easy import TrainerEasy
    from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs

    if typ == "Tuner":
        print('TrainerTuner')
        from trainers.trainer_tuner import TrainerTuner
        return TrainerTuner(config=config, *args, **kwargs)
    elif typ == "Easy":
        print('TrainerEasy')
        return TrainerEasy(config=config, *args, **kwargs)
    elif typ == "SeveralOutput":
        print('TrainerEasySeveralOutputs')
        return TrainerEasySeveralOutputs(config=config, *args, **kwargs)

    value_error("Trainer", typ)


def get_data_loader(typ: str, config, *args, **kwargs):
    from data_utils.data_loaders.data_loader import DataLoader
    from data_utils.data_loaders.data_loader_whole import DataLoaderWhole
    # from data_utils.data_loaders.archive.data_loader_colon import DataLoaderColon
    # from data_utils.data_loaders.archive.data_loader_mat import DataLoaderMat
    # from data_utils.data_loaders.archive.data_loader_mat_brain import DataLoaderMatBrain
    # from data_utils.data_loaders.archive.data_loader_mat_colon import DataLoaderMatColon
    # from data_utils.data_loaders.archive.data_loader_whole_colon import DataLoaderWholeColon
    # from data_utils.data_loaders.archive.data_loader_whole_mat import DataLoaderWholeMat
    # from data_utils.data_loaders.archive.data_loader_hno import DataLoaderHNO
    # from data_utils.data_loaders.archive.data_loader_whole_hno import DataLoaderWholeHNO

    if typ == "normal":
        return DataLoader(config=config, *args, **kwargs)
    elif typ == "whole":
        return DataLoaderWhole(config=config, *args, **kwargs)

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


def get_evaluation(labels: list, config, *args, **kwargs):
    from evaluation.evaluation_binary import EvaluationBinary
    from evaluation.evaluation_multiclass import EvaluationMulticlass

    if len(labels) == 2:
        print('Get EvaluationBinary')
        return EvaluationBinary(config=config, *args, **kwargs)
    elif len(labels) > 2:
        print('Get EvaluationMulticlass')
        return EvaluationMulticlass(config=config, *args, **kwargs)

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


def get_scaler(typ: str, config, *args, **kwargs):
    from data_utils.scaler import NormalizerScaler, StandardScalerTransposed

    if typ == 'l2_norm':
        return NormalizerScaler(config=config, *args, **kwargs)
    elif typ == 'svn':
        return StandardScalerTransposed(config=config, *args, **kwargs)
    
    value_error("scaler", typ)


def get_pixel_detection(typ: str):
    from data_utils import border

    if typ == "detect_border":
        return border.detect_border
    elif typ == "detect_core":
        return border.detect_core
    value_error("pixel detection", typ)


def get_cross_validator(typ: str, config, *args, **kwargs):
    from cross_validators.cross_validator_base import CrossValidatorBase
    from cross_validators.cross_validator_postprocessing import CrossValidatorPostProcessing

    if typ == "normal":
        return CrossValidatorBase(config=config, *args, **kwargs)
    elif typ == "spain":
        from archive.cross_validator_spain import CrossValidatorSpain
        return CrossValidatorSpain(config=config, *args, **kwargs)
    elif typ == "postprocessing":
        return CrossValidatorPostProcessing(config=config, *args, **kwargs)
    elif typ == "experiment":
        from cross_validator_experiment import CrossValidatorExperiment
        return CrossValidatorExperiment(config=config, *args, **kwargs)

    value_error("Cross validator", typ)


def get_extension_loader(typ: str, config, *args, **kwargs):
    from data_utils.data_loaders.dat_file import DatFile
    from data_utils.data_loaders.mat_file import MatFile

    if typ == ".dat":
        return DatFile(config=config, *args, **kwargs)
    elif typ == ".mat":
        return MatFile(config=config, *args, **kwargs)
    else:
        value_error(modul="file extension", typ=typ)


def get_data_storage(typ: str):
    from data_utils.data_storage import DataStorageNPZ, DataStorageZARR

    if typ == "npz":
        return DataStorageNPZ()
    elif typ == "zarr":
        return DataStorageZARR()
    else:
        value_error(modul="data storage", typ=typ)


def get_prediction_to_image(typ: str, **kwargs):
    from data_utils.prediction_to_image import PredictionToImage_npz, PredictionToImage_png
    if typ == "archive":
        return PredictionToImage_npz(**kwargs)
    elif typ == "png":
        return PredictionToImage_png(**kwargs)
    else:
        value_error(modul="prediction to image", typ=typ)


def get_dataset(typ: str, batch_size: int, d3: bool, with_sample_weights: bool, data_storage=None, dict_names=None):
    from data_utils.dataset import TFRDatasets, GeneratorDatasets
    if typ == "tfr":
        return TFRDatasets(batch_size=batch_size, d3=d3, with_sample_weights=with_sample_weights)
    elif typ == "generator":
        return GeneratorDatasets(batch_size=batch_size, d3=d3, with_sample_weights=with_sample_weights,
                                 data_storage=data_storage, dict_names=dict_names)
    else:
        value_error(modul="dataset", typ=typ)


def get_shuffle(typ: str, config, data_storage, raw_path: str, dict_names: list, set_seed: bool = True):
    from data_utils.shuffle import TFRShuffle, GeneratorShuffle
    if typ == "tfr":
        return TFRShuffle(config=config, data_storage=data_storage, raw_path=raw_path, dict_names=dict_names,
                          dataset_typ=typ, set_seed=set_seed)
    elif typ == "generator":
        return GeneratorShuffle(config=config, data_storage=data_storage, raw_path=raw_path, dict_names=dict_names,
                                dataset_typ=typ, set_seed=set_seed)
    else:
        value_error(modul="shuffle", typ=typ)


def value_error(modul: str, typ: str):
    raise ValueError(f"Error! No corresponding {modul} for {typ}")


if __name__ == '__main__':
    pass
