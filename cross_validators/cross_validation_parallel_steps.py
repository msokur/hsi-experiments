import os
import datetime
from typing import List
import sys

sys.path.insert(0, '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/')

import numpy as np
import csv
import pickle

import provider

from data_utils.dataset.choice_names import ChoiceNames
from cross_validators.cross_validator_base import CrossValidatorBase
from cross_validator_experiment_parallel import ConfigWrapper

from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK, DataLoaderKeys as DLK, \
    PreprocessorKeys as PPK, TrainerKeys as TK


def cross_validation_step(model_name: str, all_patients: List[str], leave_out_names=None):
    print(f'model_name {model_name}')
    print(f'all_patients {all_patients}')
    print(f'leave_out_names {leave_out_names}')
    
    with open(os.path.join(model_name, 'cross_validation_config.pkl'), 'rb') as file:
        config = pickle.load(file)
    with open(os.path.join(model_name, 'cross_validation_data_storage.pkl'), 'rb') as file:
        data_storage = pickle.load(file)

    if leave_out_names is None:
        leave_out_names = []
    choice_names = ChoiceNames(data_storage=data_storage, config_cv=config.CONFIG_CV,
                               labels=config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                               y_dict_name=config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1],
                               log_dir=model_name)
    valid_names = choice_names.get_valid_except_names(raw_path=config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
                                                      except_names=leave_out_names)
    train_names = list(set(all_patients) - set(leave_out_names) - set(valid_names))

    print(f"Leave out patient data: {', '.join(n for n in leave_out_names)}.\n")
    print(f"Train patient data: {', '.join(n for n in train_names)}.\n")
    print(f"Valid patient data: {', '.join(n for n in valid_names)}.\n")

    trainer = provider.get_trainer(typ=config.CONFIG_TRAINER[TK.TYPE], config=config,
                                   data_storage=data_storage, model_name=model_name,
                                   leave_out_names=leave_out_names, train_names=train_names,
                                   valid_names=valid_names)
    trainer.train()
    


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_name', type=str)
        parser.add_argument('--all_patients', type=str)
        parser.add_argument('--leave_out_names', type=str)

        args = parser.parse_args()

        print(f'Hi from CV! with {args.model_name} and {args.leave_out_names}')

        cross_validation_step(model_name=args.model_name,
                              all_patients=args.all_patients.split('+'),
                              leave_out_names=args.leave_out_names.split('+'))
        config.telegram.send_tg_message(f'CV step {args.model_name} successfully finished')

    except Exception as e:
        import configuration.get_config as config

        config.telegram.send_tg_message(f'ERROR!!!, In CV step {args.model_name} error {e}')
        raise e
