import os
import abc
import json
import itertools
import numpy as np

import config

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
        self.run_experiment()
    
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
        print()
        
class HowManyValidPatExcludeExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run_experiment(self):
        #config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)
        
        #for i, exclude in enumerate(range(4, 42, 2)):
        for i, comb in enumerate(self.combinations):
            sample_dict = {name: c for name, c in zip(self.combinations_keys, comb)}
            print(sample_dict)
            
            #combine short name
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
                        
            stream = os.popen(f'bash /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/start_cv.sh {self.root_folder} {self.name + "_" + short_name} {short_name} {i} {test_path}')
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
        'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_VALID': [*range(4, 42, 2)], 
        '_3D_SIZE': [3, 5, 7, 11]
    }
    #print(list(itertools.product()))
    HowManyValidPatExcludeExperiment('ExperimentHowManyValidPatExclude', config_for_experiment)