import os
# import datetime

# import numpy as np
# import csv

import pickle
from cross_validators.cross_validator_base import CrossValidatorBase


# from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK


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

    '''def cross_validation(self, csv_filename=None):
        name = self.config.CONFIG_CV[CVK.NAME]
        self.config.CONFIG_PATHS[PK.LOGS_FOLDER].append(name)

        root_folder = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER])
        path_template = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER], 'step')

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        paths, splits = self._get_paths_and_splits()

        csv_file = CrossValidatorBase.compose_csv_filename(csv_filename, log_dir, name)

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

            # all_patients = [self.data_storage.get_name(path=p) for p in paths] # self.cross_validation_step(
            model_name=model_name, all_patients=all_patients, leave_out_names=[self.data_storage.get_name(p) for p in 
            leave_out_paths]) 



            self.write_rows_to_csv(leave_out_paths, csv_file, log_dir, CV_step_folder)'''

    def cross_validation_step(self, trainer, dataset_paths, CV_step_name, CV_step_split):
        # all_patients = '+'.join([self.data_storage.get_name(path=p) for p in CV_step_split.all_patients])
        # test_patients = '+'.join([self.data_storage.get_name(p) for p in CV_step_split.TEST_NAMES])

        self.save_configurations(CV_step_name, CV_step_split)

        stream = os.popen(
            f"bash $HOME/Peritoneum/hsi-experiments/scripts/start_parallel_cv_step.sh {trainer.log_dir} {CV_step_name}")
        # {all_patients} "
        # f"{test_patients}")

        output = stream.read()
        print('Prompt output:', output)

    def save_configurations(self, CV_step_folder, CV_step_split):
        config_wrapper = ConfigWrapper(self.config)
        CV_config_filename = os.path.join(CV_step_folder, 'CV_config.pkl')
        with open(CV_config_filename, 'wb') as file:
            pickle.dump(config_wrapper, file)

        CV_data_storage_filename = os.path.join(CV_step_folder, 'CV_data_storage.pkl')
        with open(CV_data_storage_filename, 'wb') as file:
            pickle.dump(self.data_storage, file)

        CV_step_split_filename = os.path.join(CV_step_folder, 'CV_step_split.pkl')
        with open(CV_step_split_filename, 'wb') as file:
            pickle.dump(CV_step_split, file)
