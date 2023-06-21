import os
import abc
import json
import itertools
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import config
from evaluation.validator import Validator


class Experiment:
    def __init__(self, name, config_for_experiment):
        self.name = name
        config.MODEL_NAME_PATHS.append(self.name)
        self.root_folder = os.path.join(*config.MODEL_NAME_PATHS)
        if not os.path.exists(self.root_folder):
            os.mkdir(self.root_folder)

        self.config_for_experiment = config_for_experiment
        self.combinations_keys = config_for_experiment.keys()
        self.combinations = list(itertools.product(*config_for_experiment.values()))

        prefix = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments'
        test_root_path = os.path.join(prefix, 'test')
        test_path = os.path.join(test_root_path, name)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        self.test_path = test_path

        self.save_combinations()
        # self.run_experiment()

    def save_combinations(self):
        self.json_name = os.path.join(self.root_folder, 'combinations.json')

        result_json = []
        for comb in self.combinations:
            result_json.append({name: c for name, c in zip(self.combinations_keys, comb)})

        with open(self.json_name, 'w') as outfile:
            outfile.write(json.dumps(result_json))

    @abc.abstractmethod
    def run_experiment(self):
        return

    def get_results(self):
        folders = sorted(glob(os.path.join(self.test_path, '*/')),
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
        plt.savefig(os.path.join(self.test_path, 'means.png'))

        '''print(folder)
        checkpoints = glob(os.path.join(folder, 'cp-0000'))
        if len(checkpoints) == 0:
            print(f'WARNING!!! There is no cp-0000 for {folder}')
            continue
        checkpoint = checkpoints[0]'''


class HowManyValidPatExcludeExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_experiment(self):
        # config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)

        # for i, exclude in enumerate(range(4, 42, 2)):
        for i, comb in enumerate(self.combinations):
            sample_dict = {name: c for name, c in zip(self.combinations_keys, comb)}
            print(sample_dict)

            # combine short name
            short_name = ''
            for key, value in sample_dict.items():
                short_name += key[0]
                if 'bool' in str(type(value)):
                    if value:
                        short_name += "T"
                    else:
                        short_name += "F"
                if 'str' in str(type(value)):
                    short_name += value[0]
                if 'int' in str(type(value)):
                    short_name += str(value)
                short_name += '_'

            stream = os.popen(
                f'bash /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/start_cv.sh {self.root_folder} '
                f'{self.name + "_" + short_name} {short_name} {i} {self.test_path}')
            output = stream.read()
            print(output)


if __name__ == '__main__':
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
    config_for_experiment = {
        'WITH_SMALLER_DATASET': [False],
        'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [10],
        # [1, 3, 10, 20, 40]#[1, 2, 3, 4, 5, 6, 7, 8] + [*range(10, 42, 2)]
        # 'CV_FIRST_SPLIT': [33]
    }

    '''config_for_experiment = {
        'WITH_SMALLER_DATASET': [True],
        'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [1, 2, 3, 4, 5, 6, 7, 8] + [*range(10, 42, 2)]
    }'''

    exp = HowManyValidPatExcludeExperiment('ExperimentHowManyValidPatExclude', config_for_experiment)
    # exp.run_experiment()

    print(exp.get_results())

    '''config_for_experiment = {
        'WITH_SMALLER_DATASET': [False],
        'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [4, 10, 20]
    }
    #print(list(itertools.product()))
    HowManyValidPatExcludeExperiment('ExperimentAllDataHowManyValidPatExclude', config_for_experiment)'''
