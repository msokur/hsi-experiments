import os
import json
import itertools
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import util.experiments_combinations as utils


# TODO. Document background parameters + give warning about WITH_BACKGROUND_EXTRACTION
# TODO. Write in documentation that configuration names should be the same as in configs


class Experiment:
    def __init__(self, name, params_for_experiment, background_params=None):
        self.experiment_results_root_folder = None
        self.root_folder = None

        import configuration.get_config as config
        self.config = config
        self.name = name

        self.create_experiment_folder()

        self.parameters_for_experiment = params_for_experiment
        self.background_parameters = utils.BackgroundCombinations.validate_background_params(background_params)

        self.combinations_keys = list(params_for_experiment.keys())
        self.config_sections = [v['config_section'] for v in self.parameters_for_experiment.values()]
        self.combinations = self.create_combinations()

        print(f"Input parameters for experiment: {self.parameters_for_experiment} \n "
              f"Background params: {self.background_parameters}")
        print(f"Combinations keys: {self.combinations_keys}")
        print("Combinations:")
        print(*self.combinations, sep='\n')
        print(f"Number of combinations: {len(self.combinations)}")
        print(f"Config sections: {self.config_sections}")

        self.create_folder_for_results()

        self.save_combinations()

    def create_experiment_folder(self):
        self.root_folder = os.path.join(self.config.CONFIG_PATHS['LOGS_FOLDER'][0], self.name)
        print(f"Experiment root folder: {self.root_folder}")
        if not os.path.exists(self.root_folder):
            os.mkdir(self.root_folder)

    def create_combinations(self):
        parameters = self.get_parameters(self.parameters_for_experiment)
        combinations = list(itertools.product(*parameters))

        combinations = utils.SmoothingCombinations(self.parameters_for_experiment,
                                                   self.combinations_keys,
                                                   gaussian_params=gaussian_params,
                                                   median_params=median_params).add_combinations(combinations)
        combinations = utils.BackgroundCombinations(self.background_parameters).add_combinations(combinations)

        return combinations

    @staticmethod
    def get_parameters(parameters):
        parameters_ = []
        for value in parameters.values():
            if 'parameters' in value:
                parameters_.append(value['parameters'])
        return parameters_

    def create_folder_for_results(self):
        results_root_folder = self.config.CONFIG_PATHS['RESULTS_FOLDER']

        experiment_results_root_folder = os.path.join(results_root_folder, self.name)
        if not os.path.exists(experiment_results_root_folder):
            os.mkdir(experiment_results_root_folder)

        self.experiment_results_root_folder = experiment_results_root_folder
        print('Folder where metrics will be saved', self.experiment_results_root_folder)

    def save_combinations(self):
        self.json_name = os.path.join(self.root_folder, 'combinations.json')

        result_json = []
        for combination in self.combinations[:3]:
            print(combination)
            print(self.config_sections)
            print(self.combinations_keys)
            for section, name, value in zip(self.config_sections,
                                            self.combinations_keys,
                                            combination):
                print(section, name, value)
            print({section: [name, value] for section, name, value in zip(self.config_sections,
                                                                                       self.combinations_keys,
                                                                                       combination)})
            result_json.append({section: [name, value] for section, name, value in zip(self.config_sections,
                                                                                       self.combinations_keys,
                                                                                       combination)})
            print(result_json)

        with open(self.json_name, 'w') as outfile:
            outfile.write(json.dumps(result_json))

    def get_results(self):
        from evaluation.validator import Validator  # totally old

        folders = sorted(glob(os.path.join(self.experiment_results_root_folder, '*/')),
                         key=lambda x: int(x.split('_C')[-1].split('_')[0]))

        means_all = []
        ticks = []
        for folder in folders:
            print(folder)
            print('kaktus', folder.split(config.SYSTEM_PATHS_DELIMITER)[-2])
            tick = '_'.join(folder.split(config.SYSTEM_PATHS_DELIMITER)[-2].split('_')[1:])
            ticks.append(tick)
            print(folder)
            checkpoints = glob(os.path.join(folder, 'cp-0000'))
            if len(checkpoints) == 0:
                print(f'WARNING!!! There is no cp-0000 for {folder}')
                continue
            best_checkpoint, best_threshold, thresholds, means = Validator().find_best_checkpoint(folder)
            means_all.append(means[0])
        x = np.arange(len(means_all))
        plt.xticks(x, ticks, rotation=90)
        plt.plot(means_all)
        plt.savefig(os.path.join(self.experiment_results_root_folder, 'means.png'))

        '''print(folder)
        checkpoints = glob(os.path.join(folder, 'cp-0000'))
        if len(checkpoints) == 0:
            print(f'WARNING!!! There is no cp-0000 for {folder}')
            continue
        checkpoint = checkpoints[0]'''

    def run_experiment(self):
        for i, combination in enumerate(self.combinations):
            # print('-----------------')
            print(combination)
            #print(self.combinations_keys)
            sample_dict = {name: c for name, c in zip(self.combinations_keys, combination)}
            print(sample_dict)

            short_name = self.combine_short_name(sample_dict)
            print(short_name)

            # print(self.root_folder)
            # print(self.name + "_" + short_name)
            # print(short_name)
            # print(i)
            # print(self.experiment_results_root_folder)

            # print('-----------------')
            '''stream = os.popen(
                f'bash /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/start_cv.sh {self.root_folder} '
                f'{self.name + "_" + short_name} {short_name} {i} {self.experiment_metrics_root_folder}')
            output = stream.read()
            print(output)'''

    @staticmethod
    def combine_short_name(sample_dict):
        short_name = ''
        for key, value in sample_dict.items():
            short_name += key[0]
            if key[0].isdigit():
                short_name += key[1]

            dtype = str(type(value))
            if 'list' in dtype:
                short_name += str(value[0])
            if 'bool' in dtype:
                if value:
                    short_name += "T"
                else:
                    short_name += "F"
            if 'str' in dtype:
                short_name += value[0]
            if 'int' in dtype or 'float' in dtype:
                short_name += str(value)
            short_name += '_'

        return short_name


if __name__ == '__main__':
    gaussian_params = [0.1, 0.2]
    median_params = [3, 5, 7]

    config_for_experiment = {
        '3D_SIZE': {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': [[3, 3], [5, 5]]
        },
        "NORMALIZATION_TYPE": {
            'config_section': 'CONFIG_PREPROCESSOR',
            'parameters': ["svn_T", 'l2_norm']
        },
        "SMOOTHING_TYPE": {
            'add_None': True,
            'config_section': 'CONFIG_DATALOADER',
            'parameters': ['median_filter', 'gaussian_filter']
        },
        "SMOOTHING_VALUE": {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': median_params + gaussian_params
        },
        "BACKGROUND.WITH_BACKGROUND_EXTRACTION": {
            'config_section': 'CONFIG_DATALOADER',
            # 'parameters': [True]
        },
        "BACKGROUND.BLOOD_THRESHOLD": {
            'config_section': 'CONFIG_DATALOADER'
            # 'parameters': [0.1, 0.2, 0.3]
        },
        "BACKGROUND.LIGHT_REFLECTION_THRESHOLD": {
            'config_section': 'CONFIG_DATALOADER'
        }

    }

    background_config = {
        "BACKGROUND.WITH_BACKGROUND_EXTRACTION": {
            'parameters': [True, False]
        },
        "BACKGROUND.BLOOD_THRESHOLD": {
            'parameters': [0.1, 0.2, 0.3]
        },
        "BACKGROUND.LIGHT_REFLECTION_THRESHOLD": {
            'parameters': [0.7, 0.8]
        }
    }

    experiment = Experiment('ExperimentRevival_treeConfigs',
                            config_for_experiment,
                            background_params=background_config)
    #experiment.run_experiment()
    # print(exp.get_results())

    '''config_for_experiment = {
        '3D_SIZE':  [[3, 3], [5, 5]],
        "NORMALIZATION_TYPE": ["svn_T", 'l2_norm']
    }'''
    # ---------------------------------------------------------------------------------------------
    '''combinations = {
        'normalization': ['svn_T', 'l2_norm'],
        'patch_size': [3, 5, 7, 11],
        'smoothing': ['None', 'median', 'gaussian'],
        'gaussian_variants': [],
        'median_variatns': [],
        'with_sample_weights': [True, False],
        'with_background': [True, False],
        'background_threshold': []
    }'''

    '''config_for_experiment = {
        'WITH_SMALLER_DATASET': [True],
        'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [1, 2, 3, 4, 5, 6, 7, 8] + [*range(10, 42, 2)]
        #'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [10],
        # [1, 3, 10, 20, 40]#[1, 2, 3, 4, 5, 6, 7, 8] + [*range(10, 42, 2)]
        # 'CV_FIRST_SPLIT': [33]
    }'''

    '''config_for_experiment = {
        'WITH_SMALLER_DATASET': [False],
        'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [4, 10, 20]
    }
    #print(list(itertools.product()))
    HowManyValidPatExcludeExperiment('ExperimentAllDataHowManyValidPatExclude', config_for_experiment)'''
