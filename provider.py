import config

from trainers.trainer_easy import TrainerEasy
from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs

from data_utils.data_loaders.data_loader_easy import DataLoaderColon
from data_utils.data_loaders.data_loader_mat import DataLoaderMat
from data_utils.data_loaders.data_loader_mat_brain import DataLoaderMatBrain
from data_utils.data_loaders.data_loader_mat_colon import DataLoaderMatColon

def get_trainer(except_indexes=[]):
    if config.RESTORE_MODEL & config.WITH_TUNING:
        raise ValueError("Custom Error! Choose if restore(config.RESTORE_MODEL) or tune(config.WITH_TUNING) "
                         "model! They could not be simultaneously True")

    if config.WITH_TUNING:
        from trainers.trainer_tuner import TrainerTuner
        return TrainerTuner(excepted_indexes=except_indexes)

    if config.DATABASE == 'bea_colon':
        print('TrainerEasy')
        return TrainerEasy(excepted_indexes=except_indexes)

    if 'bea' in config.DATABASE:
        print('TrainerEasySeveralOutputs')
        return TrainerEasySeveralOutputs(excepted_indexes=except_indexes)

    return TrainerEasy(excepted_indexes=except_indexes)

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

    raise ValueError('Error! Database type specified wrong (either in config.py or in provider.py)')


if __name__ == '__main__':
    trainer = get_trainer()
    trainer.train()

    dataLoader = get_data_loader()
    dataLoader.files_read_and_save_to_npz('/work/users/mi186veva/data', '/work/users/mi186veva/data_preprocessed/raw')
    #train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])