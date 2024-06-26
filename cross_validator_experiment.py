import os
import argparse
import json
import numpy as np
from shutil import rmtree
import csv

from cross_validators.cross_validator_base import CrossValidatorBase
from data_utils.preprocessor import Preprocessor
from evaluation.optimal_parameters import OptimalThreshold



class CrossValidatorExperiment(CrossValidatorBase):
    def __init__(self, *args, **kwargs):
        try:
            self.parse_args()

            import configuration.get_config as config  # we need config from scratch every time
            self.config = config

            print(f'Hi from CV! with {self.args.experiment_folder}, {self.args.cv_name} and config_index='
                  f'{self.args.config_index}')

            #print(self.config.CONFIG_PATHS)
            #print(self.config.CONFIG_CV)
            self.config.CONFIG_CV["TYPE"] = 'experiment'
            self.config.CONFIG_CV["NAME"] = self.args.config_index + '_' + self.args.abbreviation
            self.config.CONFIG_PATHS['RESULTS_FOLDER'] = self.args.results_folder
            self.config.CONFIG_PATHS['LOGS_FOLDER'] = [self.args.experiment_folder]

            # TODO, so far could be removed, because folder is already created in  run_experiments.py
            # self.create_folder_for_results()

            self.set_configs()
            self.set_preprocessor_paths()
            self.preprocess()

            super().__init__(self.config, *args, **kwargs)
            
            #self.pipeline(execution_flags={CVK.EF_CROSS_VALIDATION: False, CVK.EF_EVALUATION: True}, thresholds=np.round(np.linspace(0.001, 0.6, 100), 4))
            self.pipeline(thresholds=np.round(np.linspace(0.001, 0.6, 100), 4))

            optimal_threshold_finder = OptimalThreshold(self.config, prints=False)
            optimal_threshold_finder.add_additional_thresholds_if_needed(self)

            rmtree(self.config.CONFIG_PATHS['RAW_NPZ_PATH'])

            self.config.telegram.send_tg_message(f'Experiment step {self.args.cv_name} (index {self.args.config_index}) is successfully completed!')
            
            self.write_status_to_csv("DONE")

        except Exception as e:
            rmtree(self.config.CONFIG_PATHS['RAW_NPZ_PATH'])

            self.config.telegram.send_tg_message(f'ERROR!!!, In experiment step {self.args.cv_name} (index {self.args.config_index}) error {e}')
            
            self.write_status_to_csv("ERROR")

            raise e
            
    def write_status_to_csv(self, status):
        row_to_add = [self.args.config_index, status, self.args.abbreviation, self.args.experiment_folder, self.args.cv_name, self.args.results_folder]

        file_path = os.path.join(self.args.experiment_folder, 'status.csv')

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_to_add)

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
            
            if '.' in config_name:  # for nested configs
                splits = config_name.split('.')
                subsection = splits[0]
                field = splits[1]
                if not subsection in config_section:
                    raise ValueError(f"{subsection} is not inside {section}! Please check names of parameters!")
                if not field in config_section[subsection]:
                    raise ValueError(f"{field} is not inside {subsection}! Please check names of parameters!")
                config_section[subsection][field] = value
            else:
                if not config_name in config_section:
                    raise ValueError(f"{config_name} is not inside {section}! Please check names of parameters!")
                config_section[config_name] = value

            #print('--------------', self.config.CONFIG_DATALOADER)
            #print('--------------', self.config.CONFIG_PREPROCESSOR)

        self.config.CONFIG_PATHS['BATCHED_PATH'] += '_' + self.args.cv_name

    def set_preprocessor_paths(self):
        self.config.CONFIG_PATHS['RAW_NPZ_PATH'] += '_' + self.args.abbreviation
        self.config.CONFIG_PATHS['SHUFFLED_PATH'] = os.path.join(self.config.CONFIG_PATHS['RAW_NPZ_PATH'], 'shuffled')
        self.config.CONFIG_PATHS['BATCHED_PATH'] = os.path.join(self.config.CONFIG_PATHS['RAW_NPZ_PATH'], 'batch_sized')

    def preprocess(self):
        preprocessor = Preprocessor(self.config)
        preprocessor.pipeline()
        

        


if __name__ == '__main__':
    CrossValidatorExperiment()
