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
    def __init__(self, name, params_for_experiment, background_params={}, replace_combinations_file=False):
        self.experiment_results_root_folder = None
        self.root_folder = None

        import configuration.get_config as config
        self.config = config
        self.name = name
        self.replace_combinations_file = replace_combinations_file

        self.create_experiment_folder()
        self.json_name = os.path.join(self.root_folder, 'combinations.json')

        self.parameters_for_experiment = params_for_experiment
        self.background_parameters = utils.BackgroundCombinations.validate_background_params(background_params)

        self.combinations_keys = list(params_for_experiment.keys())
        self.config_sections = [v['config_section'] for v in self.parameters_for_experiment.values()]
        self.combinations = self.generate_combinations()

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

    def generate_combinations(self):
        parameters = self.get_parameters(self.parameters_for_experiment)
        combinations = list(itertools.product(*parameters))
        
        
        if 'SMOOTHING.SMOOTHING_TYPE' in self.parameters_for_experiment:
            combinations = utils.SmoothingCombinations(self.parameters_for_experiment,
                                                       self.combinations_keys,
                                                       gaussian_params=gaussian_params,
                                                       median_params=median_params).add_combinations(combinations)
        if self.background_parameters:
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
        if os.path.exists(self.json_name) and not self.replace_combinations_file:
            raise ValueError(f"{self.json_name} already exists! You can set replace_combinations_file to True, but "
                             f"be careful to not to loose combinations in the existing file")

        result_json = []
        for combination in self.combinations:
            result_json.append({name: [section, value] for section, name, value in zip(self.config_sections,
                                                                                       self.combinations_keys,
                                                                                       combination)})
        with open(self.json_name, 'w') as outfile:
            outfile.write(json.dumps(result_json))

    
    def run_combination(self, combination, i):
        print('combination', combination)
        #print(self.combinations_keys)
        sample_dict = {name: c for name, c in zip(self.combinations_keys, combination)}
        print('sample_dict', sample_dict)

        short_name = self.combine_short_name(sample_dict)
        print('short_name', short_name)

        print('self.root_folder', self.root_folder)
        print('self.name + "_" + short_name', self.name + "_" + short_name)
        print('short_name', short_name)
        print('Index', i)
        print('self.experiment_results_root_folder', self.experiment_results_root_folder)
        
        parallel = ''
        if PARALLEL:
            parallel = '_parallel'
        
        execution = f'bash /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/start_cv{parallel}.sh {self.root_folder} {self.name + "_" + short_name} {short_name} {i} {self.experiment_results_root_folder} {self.name}'
        
        print(execution)
        stream = os.popen(execution)
        output = stream.read()
        print('Prompt output:',output)
        print('--------------------------------------------------------------------------------------------------')
    
   
            
    def run_experiment_normal(self, combinations=None):
        if combinations is None:
            combinations = [(i, combination) for i, combination in enumerate(self.combinations)][1:2]
            #combinations = list(np.array([(i, combination) for i, combination in enumerate(self.combinations)])[[24]])
        #for i in range(29, 30):
        for i, combination in combinations:
            # print('-----------------')
            #if i == 28:
            if True:
                self.run_combination(combination, i)
         
    
    def run_experiment_schedule(self):
        #self.combinations = list(np.array([(i, combination) for i, combination in enumerate(self.combinations)])[[201, 214, 215, 228, 234]])
        self.combinations = [(i, combination) for i, combination in enumerate(self.combinations)]
        import schedule
        import time
        import threading


        def job():
            if self.combinations:
                combinations_batch = self.combinations[:5]
                self.combinations = self.combinations[5:]
                threading.Thread(target=self.run_experiment_normal, args=(combinations_batch,)).start()


        schedule.every(1).hours.do(job)

        job()

        # Основний цикл для перевірки планувальника
        while self.combinations:
            schedule.run_pending()
            time.sleep(1)

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
    PARALLEL = True
    import time
    #time.sleep(5 * 60 * 60)

    #gaussian_params = [0.5, 1, 1.5]   #for size 3 and 5, smoothing 1d and 3d
    #gaussian_params = [1, 2, 3]      #for size 3 and 5, smoothing 2d 
    #median_params = [3, 5, 7]   #for size 3 and 5, all smoothing dimentions
    
    #gaussian_params = [0.5]   #for size 7, smoothing 1d
    #median_params = [3, 5]   #for size 7, smoothing 1d
    
    gaussian_params = [2]   #for size 7, smoothing 2d
    median_params = [3]   #for size 7, smoothing 2d
 
    config_for_experiment = {
        '3D_SIZE': {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': [[7, 7]] #[[5, 5]]#, [7, 7], [11, 11]]
        },
        "NORMALIZATION_TYPE": {
           'config_section': 'CONFIG_PREPROCESSOR',
            'parameters': ["svn"]
        },
        "WITH_SAMPLE_WEIGHTS": {
            'config_section': 'CONFIG_TRAINER',
            'parameters': [True]
        },
        "SMOOTHING.SMOOTHING_DIMENSIONS": {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': ['2d']
        },
        "SMOOTHING.SMOOTHING_TYPE": {
            'add_None': True,
            'config_section': 'CONFIG_DATALOADER',
            'parameters': ['median_filter', 'gaussian_filter']
        },
        "SMOOTHING.SMOOTHING_VALUE": {
            'config_section': 'CONFIG_DATALOADER',
            'parameters': np.unique(median_params + gaussian_params).astype(float)
        },
        "BACKGROUND.WITH_BACKGROUND_EXTRACTION": {
            'config_section': 'CONFIG_DATALOADER',
        },
        "BACKGROUND.BLOOD_THRESHOLD": {
            'config_section': 'CONFIG_DATALOADER'
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
            'parameters': [0.1]
        },
        "BACKGROUND.LIGHT_REFLECTION_THRESHOLD": {
            'parameters': [0.25, 0.4]
        }
    }

    experiment = Experiment('WHOLE_MainExperiment_7_smoothing_2d',
                            config_for_experiment,
                            background_params=background_config,
                            replace_combinations_file=True)
    experiment.run_experiment_normal()
    # print(exp.get_results())
