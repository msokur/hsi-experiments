import os
import argparse
import json
import numpy as np
from shutil import rmtree

from cross_validators.cross_validator_base import CrossValidatorBase
from data_utils.preprocessor import Preprocessor


class CrossValidatorExperiment(CrossValidatorBase):
    def __init__(self, *args, **kwargs):
        self.parse_args()

        import configuration.get_config as config  # we need config from scratch every time
        self.config = config

        print(f'Hi from CV! with {self.args.experiment_folder}, {self.args.cv_name} and config_index='
              f'{self.args.config_index}')

        print(self.config.CONFIG_PATHS)
        print(self.config.CONFIG_CV)
        self.config.CONFIG_CV["TYPE"] = 'experiment'
        self.config.CONFIG_CV["NAME"] = self.args.abbreviation
        self.config.CONFIG_PATHS['RESULTS_FOLDER'] = self.args.results_folder
        self.config.CONFIG_PATHS['LOGS_FOLDER'] = [self.args.experiment_folder]

        # TODO, so far could be removed, because folder is already created in  run_experiments.py
        # self.create_folder_for_results()

        self.set_configs()
        self.set_preprocessor_paths()
        self.generate_data()

        super().__init__(self.config, *args, **kwargs)

        self.pipeline(thresholds=np.round(np.linspace(0.1, 0.5, 5), 4))

        rmtree(self.config.CONFIG_PATHS['RAW_NPZ_PATH'])

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Process some integers.')

        parser.add_argument('--experiment_folder', type=str)
        parser.add_argument('--cv_name', type=str)
        parser.add_argument('--config_index', type=str)
        parser.add_argument('--results_folder', type=str)
        parser.add_argument('--abbreviation', type=str)

        args = parser.parse_args()
        self.args = args
        print('Parsed args:', self.args)

    def create_folder_for_results(self):
        results_folder = os.path.join(self.args.results_folder, self.args.abbreviation)
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        self.results_folder = results_folder
        print(f"Folder for results: {self.results_folder}")

    def set_configs(self):
        configs = None
        with open(os.path.join(self.args.experiment_folder, 'combinations.json'), 'r') as json_file:
            data = json.load(json_file)
            configs = data[int(self.args.config_index)]
            print(configs)

        print(configs)
        for config_name, params in configs.items():
            print(config_name, params)
            section, value = params
            config_section = getattr(self.config, section)
            config_section[config_name] = value

            #print('--------------', self.config.CONFIG_DATALOADER)
            #print('--------------', self.config.CONFIG_PREPROCESSOR)
        #config.CV_RESTORE_VALID_PATIENTS_SEQUENCE = self.args.abbreviation.replace('WF', 'WT') # TODO

        self.config.CONFIG_PATHS['BATCHED_PATH'] += '_' + self.args.cv_name

    def set_preprocessor_paths(self):
        self.config.CONFIG_PATHS['RAW_NPZ_PATH'] += '_' + self.args.abbreviation
        self.config.CONFIG_PATHS['SHUFFLED_PATH'] = os.path.join(self.config.CONFIG_PATHS['RAW_NPZ_PATH'], 'shuffled')
        self.config.CONFIG_PATHS['BATCHED_PATH'] = os.path.join(self.config.CONFIG_PATHS['RAW_NPZ_PATH'], 'batch_sized')

    def generate_data(self):
        preprocessor = Preprocessor(self.config)
        preprocessor.pipeline()


if __name__ == '__main__':
    CrossValidatorExperiment()
