import config

from trainers.trainer_easy import TrainerEasy
from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs


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

    


if __name__ == '__main__':
    trainer = get_trainer()
    trainer.train()

    #train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
