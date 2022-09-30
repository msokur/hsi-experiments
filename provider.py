import config


from data_utils.data_loaders.data_loader_colon import DataLoaderColon
from data_utils.data_loaders.data_loader_mat import DataLoaderMat
from data_utils.data_loaders.data_loader_mat_brain import DataLoaderMatBrain
from data_utils.data_loaders.data_loader_mat_colon import DataLoaderMatColon
from data_utils.data_loaders.data_loader_whole_colon import DataLoaderWholeColon
from data_utils.data_loaders.data_loader_whole_mat import DataLoaderWholeMat
from data_utils.smoothing import MedianFilter, GaussianFilter
import trainer_easy
import trainer_easy_several_outputs
from evaluation.evaluation_binary import EvaluationBinary
from evaluation.evaluation_multiclass import EvaluationMulticlass
#import models.keras_tuner_model as keras_tuner_model
#import models.keras_tuner_models_with_ones as keras_tuner_models_with_ones


def get_trainer(*args, **kwargs):
    if config.RESTORE_MODEL & config.WITH_TUNING:
        raise ValueError("Custom Error! Choose if restore(config.RESTORE_MODEL) or tune(config.WITH_TUNING) "
                         "model! They could not be simultaneously True")

    if config.WITH_TUNING:
        from trainers.trainer_tuner import TrainerTuner
        print('TrainerTuner')
        return TrainerTuner(*args, **kwargs)

    if config.DATABASE == 'bea_colon':
        print('TrainerEasy')
        return trainer_easy.TrainerEasy(*args, **kwargs)

    if 'bea' in config.DATABASE:
        print('TrainerEasySeveralOutputs')
        return trainer_easy_several_outputs.TrainerEasySeveralOutputs(*args, **kwargs)


    return trainer_easy.TrainerEasy(*args, **kwargs)


def get_data_loader(**kwargs):
    if config.DATABASE == 'colon':
        print('DataLoaderColon')
        return DataLoaderColon(**kwargs)
    if config.DATABASE == 'bea_eso':
        print('DataLoaderMat')
        return DataLoaderMat(**kwargs)
    if config.DATABASE == 'bea_brain':
        print('DataLoaderMatBrain')
        return DataLoaderMatBrain(**kwargs)
    if config.DATABASE == 'bea_colon':
        print('DataLoaderMatColon')
        return DataLoaderMatColon(**kwargs)

    # analogs for whole cubes
    if config.DATABASE == 'colon_whole':
        print('DataLoaderWholeColon')
        return DataLoaderWholeColon(**kwargs)
    if config.DATABASE == 'bea_brain_whole':
        print('DataLoaderWholeMat with Brain')
        return DataLoaderWholeMat(DataLoaderMatBrain(**kwargs), **kwargs)
    if config.DATABASE == 'bea_eso_whole':
        print('DataLoaderWholeMat with Eso')
        return DataLoaderWholeMat(DataLoaderMat(**kwargs), **kwargs)
    if config.DATABASE == 'bea_colon_whole':
        print('DataLoaderWholeMat with Colon')
        return DataLoaderWholeMat(DataLoaderMatColon(**kwargs), **kwargs)

    raise ValueError(f'Error! Database type {config.DATABASE} specified wrong (either in config.py or in provider.py)')


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

    if analog == "":
        raise ValueError(f"We didn't found an analog whole database for {original_database}")

    print(f"For {original_database} we have found {analog} whole database")

    return analog


def get_evaluation(*args, **kwargs):
    labels = config.LABELS_OF_CLASSES_TO_TRAIN
    print('labels', labels)
    if len(labels) == 2:
        print('Get EvaluationBinary')
        return EvaluationBinary(*args, **kwargs)
    print('Get EvaluationMulticlass')
    return EvaluationMulticlass(*args, **kwargs)


def get_keras_tuner_model():
    if config.TUNER_MODEL == 'KerasTunerModel':
        return keras_tuner_model.KerasTunerModel()
    if config.TUNER_MODEL == 'KerasTunerModelOnes':
        return keras_tuner_models_with_ones.KerasTunerModelOnes()

    raise ValueError(f'Error! Tuner model type {config.TUNER_MODEL} specified wrong (either in config.py or in '
                     f'provider.py)')

def get_smoother(*args, **kwargs):
    if config.SMOOTHING_TYPE == 'median_filter':
        return MedianFilter(*args, **kwargs)
    if config.SMOOTHING_TYPE == 'gaussian_filter':
        return GaussianFilter(*args, **kwargs)
    
    raise ValueError(f'Error! Smoother type is {config.SMOOTHING_TYPE}')

if __name__ == '__main__':
    trainer = get_trainer(except_indexes=['2020_02_04_20_48_03_'], valid_except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
    trainer.train()

    #dataLoader = get_data_loader()
    #dataLoader.files_read_and_save_to_npz('/work/users/mi186veva/data', '/work/users/mi186veva/data_preprocessed/raw')
    #train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
