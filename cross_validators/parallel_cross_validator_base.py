import os

import pickle
from cross_validators.cross_validator_base import CrossValidatorBase
from configuration.keys import CrossValidationKeys as CVK


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

    def cross_validation_step(self, trainer, dataset_paths, CV_step_name, CV_step_split):
        self.execution_flags[CVK.EF_EVALUATION] = False # parallel CV is used to start jobs, it's not possible to start evaluation immideately, because we should wait until this jobs finish
        dataset_paths = '+'.join(dataset_paths)

        self.save_configurations(os.path.join(trainer.log_dir, CV_step_name), CV_step_split)

        stream = os.popen(
            f"sbatch {os.path.join(os.getcwd(), 'start_parallel_cv_step.job')} {trainer.log_dir} {CV_step_name} {dataset_paths} {self.config.CONFIG_CV[CVK.NAME]}")

        output = stream.read()
        print('Prompt output:', output)

    def save_configurations(self, CV_step_folder , CV_step_split):
        if not os.path.exists(CV_step_folder):
            os.mkdir(CV_step_folder)
        
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
