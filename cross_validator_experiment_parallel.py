import os
import argparse
import datetime
import json
import numpy as np
from shutil import rmtree, copy
import csv
import importlib.util
import pickle

from cross_validator_experiment import CrossValidatorExperiment
from data_utils.preprocessor import Preprocessor
from evaluation.optimal_parameters import OptimalThreshold
from configuration.keys import CrossValidationKeys as CVK
import provider
from configuration.parameter import STORAGE_TYPE
from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK

CV = True

class ConfigWrapper:
    def __init__(self, config_module):
        self.CONFIG_CV = config_module.CONFIG_CV
        self.CONFIG_DATALOADER = config_module.CONFIG_DATALOADER
        self.CONFIG_PATHS = config_module.CONFIG_PATHS
        self.CONFIG_TRAINER = config_module.CONFIG_TRAINER
        self.CONFIG_PREPROCESSOR = config_module.CONFIG_PREPROCESSOR


class CrossValidatorExperimentParallel(CrossValidatorExperiment):
    def __init__(self, *args, **kwargs):
        try:            
            self.parse_args()

            import configuration.get_config as config  # we need config from scratch every time
            self.config = config
            self.data_storage = provider.get_data_storage(typ=STORAGE_TYPE)

            print(f'Hi from Experiment Parallel! with {self.args.experiment_folder}, {self.args.cv_name} and config_index='
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

            #super().__init__(self.config, *args, **kwargs)
            
            if CV:
                self.preprocess()
                self.pipeline(execution_flags={CVK.EF_CROSS_VALIDATION: True, CVK.EF_EVALUATION: False}, thresholds=np.round(np.linspace(0.001, 0.6, 100), 4))
            else:
                self.pipeline(execution_flags={CVK.EF_CROSS_VALIDATION: False, CVK.EF_EVALUATION: True}, thresholds=np.round(np.linspace(0.001, 0.6, 100), 4))
                #self.pipeline(thresholds=np.round(np.linspace(0.001, 0.6, 100), 4))

                optimal_threshold_finder = OptimalThreshold(self.config, prints=False)
                optimal_threshold_finder.add_additional_thresholds_if_needed(self)

                #rmtree(self.config.CONFIG_PATHS['RAW_NPZ_PATH'])

            self.config.telegram.send_tg_message(f'Experiment step {self.args.cv_name} (index {self.args.config_index}) is successfully completed!')
            
            self.write_status_to_csv("DONE")

        except Exception as e:
            #rmtree(self.config.CONFIG_PATHS['RAW_NPZ_PATH'])

            self.config.telegram.send_tg_message(f'ERROR!!!, In experiment step {self.args.cv_name} (index {self.args.config_index}) error {e}')
            
            self.write_status_to_csv("ERROR")

            raise e
            
            
    def cross_validation(self, csv_filename=None):
        name = self.config.CONFIG_CV[CVK.NAME]
        self.config.CONFIG_PATHS[PK.LOGS_FOLDER].append(name)

        root_folder = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER])
        path_template = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER], 'step')

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        paths, splits = self._get_paths_and_splits()

        date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

        if csv_filename is None:
            csv_filename = os.path.join(root_folder, name + "_stats" + date_ + ".csv")
        else:
            csv_filename = os.path.join(root_folder, csv_filename)

        for indexes in splits[self.config.CONFIG_CV[CVK.FIRST_SPLIT]:]:
            model_name = path_template
            if len(indexes) > 1:
                for i in indexes:
                    model_name += "_" + str(i)
            else:
                model_name += "_" + str(indexes[0]) + "_" + self.data_storage.get_name(np.array(paths)[indexes][0])

            leave_out_paths = np.array(paths)[indexes]

            if self.__check_data_label__(leave_out_paths):
                print(f"The patient file(s) '{', '.join(leave_out_paths)}' are no needed labels for training! "
                      f"So we skip this patient(s)!")
                continue
                
            if not os.path.exists(model_name):
                os.mkdir(model_name)

            all_patients = '+'.join([self.data_storage.get_name(path=p) for p in paths])
            leave_out_names = '+'.join([self.data_storage.get_name(p) for p in leave_out_paths])
            
            config_wrapper = ConfigWrapper(self.config)
            CV_config_filename = os.path.join(model_name, 'cross_validation_config.pkl')
            with open(CV_config_filename, 'wb') as file:
                pickle.dump(config_wrapper, file)
                
            CV_data_storage_filename = os.path.join(model_name, 'cross_validation_data_storage.pkl')
            with open(CV_data_storage_filename, 'wb') as file:
                pickle.dump(self.data_storage, file)
                
            experiment = model_name.split(self.config.CONFIG_PATHS[PK.SYS_DELIMITER])[-3]
            combination = model_name.split(self.config.CONFIG_PATHS[PK.SYS_DELIMITER])[-2]
            step = model_name.split(self.config.CONFIG_PATHS[PK.SYS_DELIMITER])[-1]

            stream = os.popen(
                f'bash /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/start_cv_step.sh '
                f'{model_name} {all_patients} {leave_out_names} {experiment} {combination} {step}')
            output = stream.read()
            print('Prompt output:', output)
            # self.cross_validation_step(model_name=model_name, all_patients=all_patients,
            #                           leave_out_names=[self.data_storage.get_name(p) for p in leave_out_paths])

            for i, path_ in enumerate(leave_out_paths):
                sensitivity, specificity = 0, 0
                with open(csv_filename, 'a', newline='') as csvfile:  # for full cross_valid and for separate file
                    fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'index': str(i),
                                     'sensitivity': str(sensitivity),
                                     'specificity': str(specificity),
                                     'name': path_,
                                     'model_name': model_name})
        

        


if __name__ == '__main__':
    CrossValidatorExperimentParallel()
