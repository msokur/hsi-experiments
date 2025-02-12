import os
import datetime
from typing import List
import sys
from pprint import pprint

import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pickle
import provider


from configuration.keys import TrainerKeys as TK, PathKeys as PK


def cross_validation_step(CV_folder, CV_step_name, dataset_paths):
    CV_step_folder = os.path.join(CV_folder, CV_step_name)
    print(f'CV_step_folder {CV_step_folder}')

    with open(os.path.join(CV_step_folder, 'CV_config.pkl'), 'rb') as file:
        config = pickle.load(file)
    with open(os.path.join(CV_step_folder, 'CV_data_storage.pkl'), 'rb') as file:
        data_storage = pickle.load(file)
    with open(os.path.join(CV_step_folder, 'CV_step_split.pkl'), 'rb') as file:
        CV_step_split = pickle.load(file)

    trainer = provider.get_trainer(typ=config.CONFIG_TRAINER[TK.TYPE],
                                   config=config,
                                   data_storage=data_storage,
                                   log_dir=CV_folder)

    trainer.train(cv_step_split=CV_step_split,
                  step_name=CV_step_name,
                  dataset_paths=dataset_paths,
                  batch_path=os.path.join(config.CONFIG_PATHS[PK.BATCHED_PATH], CV_step_name))


if __name__ == '__main__':
    from configuration.get_config import CVConfig
    Config = CVConfig()
    args = None
    try:   
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--CV_folder', type=str)
        parser.add_argument('--CV_step_name', type=str)
        parser.add_argument('--dataset_paths', type=str)

        args = parser.parse_args()
        dataset_paths = args.dataset_paths.split('+')

        print(f'Hi from parallel CV {args.CV_folder} with {args.CV_step_name}!')
        pprint(f'dataset paths {dataset_paths}')
    
        cross_validation_step(CV_folder=args.CV_folder, 
                              CV_step_name=args.CV_step_name,
                              dataset_paths=dataset_paths)
        Config.telegram.send_tg_message(f'CV step {args.CV_folder} successfully finished')

    except Exception as e:
        if args:
            Config.telegram.send_tg_message(f'ERROR!!!, In CV step {args.CV_folder} error {e}')
        else:
            Config.telegram.send_tg_message(f'ERROR!!!, args parsing was probably not successful in CV step, error {e}')
        raise e
