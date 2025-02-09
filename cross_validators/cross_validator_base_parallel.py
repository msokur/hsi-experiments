import os
import datetime

import numpy as np
import csv

import pickle
from cross_validators.cross_validator_base import CrossValidatorBase
from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK


class ConfigWrapper:
    def __init__(self, config_module):
        self.CONFIG_CV = config_module.CONFIG_CV
        self.CONFIG_DATALOADER = config_module.CONFIG_DATALOADER
        self.CONFIG_PATHS = config_module.CONFIG_PATHS
        self.CONFIG_TRAINER = config_module.CONFIG_TRAINER
        self.CONFIG_PREPROCESSOR = config_module.CONFIG_PREPROCESSOR


class CrossValidatorBaseParallel(CrossValidatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                if not os.path.exists(model_name):
                    os.makedirs(model_name)

            leave_out_paths = np.array(paths)[indexes]

            if self.__check_data_label__(leave_out_paths):
                print(f"The patient file(s) '{', '.join(leave_out_paths)}' are no needed labels for training! "
                      f"So we skip this patient(s)!")
                continue

            # all_patients = [self.data_storage.get_name(path=p) for p in paths]
            # self.cross_validation_step(model_name=model_name, all_patients=all_patients, leave_out_names=[self.data_storage.get_name(p) for p in leave_out_paths])

            all_patients = '+'.join([self.data_storage.get_name(path=p) for p in paths])
            leave_out_names = '+'.join([self.data_storage.get_name(p) for p in leave_out_paths])

            config_wrapper = ConfigWrapper(self.config)
            CV_config_filename = os.path.join(model_name, 'cross_validation_config.pkl')
            with open(CV_config_filename, 'wb') as file:
                pickle.dump(config_wrapper, file)

            CV_data_storage_filename = os.path.join(model_name, 'cross_validation_data_storage.pkl')
            with open(CV_data_storage_filename, 'wb') as file:
                pickle.dump(self.data_storage, file)

            stream = os.popen(
                f"bash $HOME/Peritoneum/hsi-experiments/scripts/start_cv_step.sh {model_name} {all_patients} {leave_out_names}")

            output = stream.read()
            print('Prompt output:', output)

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
