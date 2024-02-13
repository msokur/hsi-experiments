import os
import json
import itertools
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, name, parameters_for_experiment):
        self.experiment_results_root_folder = None
        self.root_folder = None

        import configuration.get_config as config
        self.config = config
        self.name = name

        self.create_experiment_folder()

        self.parameters_for_experiment = parameters_for_experiment
        self.combinations_keys = list(parameters_for_experiment.keys())
        self.config_sections = [v['config_section'] for v in self.parameters_for_experiment.values()]
        self.combinations = self.create_combinations()
        print(f"Input parameters for experiment: {self.parameters_for_experiment}")
        print(f"Combinations keys: {self.combinations_keys}")
        print(f"Combinations: {self.combinations}")
        print(f"Config sections: {self.config_sections}")

        self.create_folder_for_results()

        self.save_combinations()

    def create_experiment_folder(self):
        self.root_folder = os.path.join(self.config.CONFIG_PATHS['LOGS_FOLDER'][0], self.name)
        print(f"Experiment root folder: {self.root_folder}")
        if not os.path.exists(self.root_folder):
            os.mkdir(self.root_folder)

    def filter_background_combinations(self, combinations):
        if 'BACKGROUND.WITH_BACKGROUND_EXTRACTION' in self.combinations_keys and \
                ('BACKGROUND.BLOOD_THRESHOLD' in self.combinations_keys or
                 'BACKGROUND.LIGHT_REFLECTION_THRESHOLD' in self.combinations_keys):
            with_background_extraction_index = self.combinations_keys.index('BACKGROUND.WITH_BACKGROUND_EXTRACTION')

            appeared_BackgroundExtractionFalse = False
            filtered_combinations = []
            for combination in combinations:
                if combination[with_background_extraction_index]:
                    filtered_combinations.append(combination)
                    continue
                if (not combination[with_background_extraction_index]) and (not appeared_BackgroundExtractionFalse):
                    appeared_BackgroundExtractionFalse = True
                    filtered_combinations.append(combination)

            return filtered_combinations
        return combinations

    def create_combinations(self):
        parameters = [v['parameters'] for v in self.parameters_for_experiment.values()]
        print('parameters', parameters)
        combinations = list(itertools.product(*parameters))

        combinations = self.filter_background_combinations(combinations)
        combinations = self.filter_smoothing_values(combinations)

        return combinations

    def filter_smoothing_values(self, combinations):
        if 'SMOOTHING_TYPE' in self.combinations_keys and 'SMOOTHING_VALUE' in self.combinations_keys:
            smoothing_type_index = self.combinations_keys.index('SMOOTHING_TYPE')
            smoothing_value_index = self.combinations_keys.index('SMOOTHING_VALUE')

            filtered_combinations = []
            appeared_SmoothingNone = False
            for combination in combinations:
                if combination[smoothing_type_index] == 'gaussian_filter' and \
                        combination[smoothing_value_index] in gaussian_params:
                    filtered_combinations.append(combination)
                if combination[smoothing_type_index] == 'median_filter' and \
                        combination[smoothing_value_index] in median_params:
                    filtered_combinations.append(combination)

                if combination[smoothing_type_index] is None and (not appeared_SmoothingNone):
                    appeared_SmoothingNone = True
                    filtered_combinations.append(combination)
            return filtered_combinations
        return combinations


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
        for combination in self.combinations:
            result_json.append({section: [name, value] for section, name, value in zip(self.config_sections,
                                                                                       self.combinations_keys,
                                                                                       combination)})

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
            #print('-----------------')
            print(combination)
            sample_dict = {name: c for name, c in zip(self.combinations_keys, combination)}
            #print(sample_dict)

            short_name = self.combine_short_name(sample_dict)
            #print(short_name)


            #print(self.root_folder)
            #print(self.name + "_" + short_name)
            #print(short_name)
            #print(i)
            #print(self.experiment_results_root_folder)

            #print('-----------------')
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
        "BACKGROUND.WITH_BACKGROUND_EXTRACTION": {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': [True]
        },
        "BACKGROUND.BLOOD_THRESHOLD": {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': [0.1, 0.2, 0.3]
        },
        #"SMOOTHING_TYPE": {
        #    'config_section': 'CONFIG_DATALOADER',
        #    'parameters': [None, 'median_filter', 'gaussian_filter']
        #},
        #"SMOOTHING_VALUE": {
        #    'config_section': 'CONFIG_DATALOADER',
        #    'parameters': median_params + gaussian_params
        #},
    }

    experiment = Experiment('ExperimentRevival_treeConfigs', config_for_experiment)
    experiment.run_experiment()
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
