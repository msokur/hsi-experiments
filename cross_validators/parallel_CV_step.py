import os
import datetime
from typing import List
import sys

import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

#import numpy as np
#import csv
import pickle

import provider

#from data_utils.dataset.choice_names import ChoiceNames
#from cross_validators.cross_validator_base import CrossValidatorBase
#from cross_validator_experiment_parallel import ConfigWrapper

from configuration.keys import TrainerKeys as TK, PathKeys as PK
#, CrossValidationKeys as CVK,  DataLoaderKeys as DLK, PreprocessorKeys as PPK


def cross_validation_step(CV_folder, CV_step_name):#, #all_patients: List[str], leave_out_names=None):
    CV_step_folder = os.path.join(CV_folder, CV_step_name)
    print(f'CV_step_folder {CV_step_folder}')
    #print(f'all_patients {all_patients}')
    #print(f'leave_out_names {leave_out_names}')

    with open(os.path.join(CV_step_folder, 'CV_config.pkl'), 'rb') as file:
        config = pickle.load(file)
    with open(os.path.join(CV_step_folder, 'CV_data_storage.pkl'), 'rb') as file:
        data_storage = pickle.load(file)
    with open(os.path.join(CV_step_folder, 'CV_step_split.pkl'), 'rb') as file:
        CV_step_split = pickle.load(file)

    #if leave_out_names is None:
    #    leave_out_names = []
    #choice_names = ChoiceNames(data_storage=data_storage, config_cv=config.CONFIG_CV,
    #                           labels=config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
    #                           y_dict_name=config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1],
    #                           log_dir=CV_step_folder)
    #valid_names = choice_names.get_valid_except_names(raw_path=config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
    #                                                  except_names=leave_out_names)
    #train_names = list(set(all_patients) - set(leave_out_names) - set(valid_names))

    #print(f"Leave out patient data: {', '.join(n for n in leave_out_names)}.\n")
    #print(f"Train patient data: {', '.join(n for n in train_names)}.\n")
    #print(f"Valid patient data: {', '.join(n for n in valid_names)}.\n")

    '''trainer = provider.get_trainer(typ=config.CONFIG_TRAINER[TK.TYPE], config=config,
                                   data_storage=data_storage, CV_step_folder=CV_step_folder,
                                   leave_out_names=leave_out_names, train_names=train_names,
                                   valid_names=valid_names)'''

    trainer = provider.get_trainer(typ=config.CONFIG_TRAINER[TK.TYPE],
                                   config=config,
                                   data_storage=data_storage,
                                   log_dir=CV_folder)

    trainer.train(cv_step_split=CV_step_split,
                  step_name=CV_step_name,
                  batch_path=os.path.join(config.CONFIG_PATHS[PK.BATCHED_PATH], CV_step_name))


if __name__ == '__main__':
    try:
        import configuration.get_config as config
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument('--CV_folder', type=str)
        parser.add_argument('--CV_step_name', type=str)
        #parser.add_argument('--all_patients', type=str)
        #parser.add_argument('--leave_out_names', type=str)

        args = parser.parse_args()

        #print(f'Hi from parallel CV! with {args.CV_step_folder} and {args.leave_out_names}')
        print(f'Hi from parallel CV {args.CV_step_folder} with {args.CV_step_name}!')

        cross_validation_step(CV_folder=args.CV_folder, CV_step_name=args.CV_step_name)
                              #all_patients=args.all_patients.split('+'),
                              #leave_out_names=args.leave_out_names.split('+'))
        config.telegram.send_tg_message(f'CV step {args.CV_step_folder} successfully finished')

    except Exception as e:

        config.telegram.send_tg_message(f'ERROR!!!, In CV step {args.CV_step_folder} error {e}')
        raise e
