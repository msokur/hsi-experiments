import os
from glob import glob
import datetime
from typing import List

import numpy as np
import csv

from util import utils
from util.compare_distributions import DistributionsChecker
import provider

from data_utils.paths import get_sort, get_splits
from trainers import CVStepNames, Trainer

from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK, DataLoaderKeys as DLK, \
    PreprocessorKeys as PPK, TrainerKeys as TK
from configuration.parameter import (
    STORAGE_TYPE,
)


class CrossValidatorBase:
    def __init__(self, config):
        self.config = config
        self.data_storage = provider.get_data_storage(typ=STORAGE_TYPE)
        self.log_root_folder = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER])

    @staticmethod
    def get_execution_flags():
        return {
            CVK.EF_CROSS_VALIDATION: True,
            CVK.EF_EVALUATION: True
        }

    def pipeline(self, execution_flags=None, **kwargs):
        if execution_flags is None:
            execution_flags = CrossValidatorBase.get_execution_flags()

        csv_filename = None
        if CVK.CSV_FILENAME in self.config.CONFIG_CV:
            csv_filename = self.config.CONFIG_CV[CVK.CSV_FILENAME]

        if execution_flags[CVK.EF_CROSS_VALIDATION]:
            self.cross_validation(csv_filename=csv_filename)
        if execution_flags[CVK.EF_EVALUATION]:
            self.evaluation(**kwargs)

    def evaluation(self, save_predictions=True, **kwargs):
        evaluator = provider.get_evaluation(config=self.config,
                                            labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN])

        if save_predictions:
            training_csv_path = self.get_csv(os.path.join(self.config.CONFIG_PATHS[PK.LOGS_FOLDER][0],
                                                          self.config.CONFIG_CV[CVK.NAME]))
            print('training_csv_path', training_csv_path)
            evaluator.save_predictions_and_metrics(training_csv_path=training_csv_path,
                                                   data_folder=self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
                                                   **kwargs)
        else:
            evaluator.evaluate(**kwargs)

    def cross_validation_step(self, trainer: Trainer, dataset_paths: List[str], train_step_name: str,
                              cv_step_names: CVStepNames):
        cv_step_names.print_names()
        trainer.train(dataset_paths=dataset_paths,
                      train_step_names=cv_step_names,
                      step_name=train_step_name,
                      batch_path=os.path.join(self.config.CONFIG_PATHS[PK.BATCHED_PATH], train_step_name))

    def cross_validation(self, csv_filename=None):
        root_folder = str(os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER]))

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        paths, splits = self._get_paths_and_splits()

        name = self.config.CONFIG_CV[CVK.NAME]
        log_dir = os.path.join(root_folder, name)
        all_patients_names = [self.data_storage.get_name(path=p) for p in paths]
        trainer = provider.get_trainer(typ=self.config.CONFIG_TRAINER[TK.TYPE],
                                       config=self.config,
                                       data_storage=self.data_storage,
                                       log_dir=log_dir)
        cv_step_names = CVStepNames(data_storage=self.data_storage,
                                    config=self.config,
                                    log_dir=log_dir,
                                    all_patients=all_patients_names)

        dataset_paths = trainer.get_dataset_paths()

        if self.config.CONFIG_TRAINER[TK.TYPE] == "Tuner" or self.config.CONFIG_TRAINER[TK.USE_SMALLER_DATASET]:
            print("--- Searching for representative smaller dataset ---")
            dc = DistributionsChecker(paths=dataset_paths,
                                      dataset=trainer.dataset,
                                      local_config=self.config.CONFIG_DISTRIBUTION)
            tuning_index = dc.get_small_database_for_tuning()

            if self.config.CONFIG_TRAINER[TK.TYPE] == "Tuner":
                trainer.search_for_hyper_parameter(tuning_data_paths=[dataset_paths[tuning_index]],
                                                   patient_names=all_patients_names)

            if self.config.CONFIG_TRAINER[TK.USE_SMALLER_DATASET]:
                dataset_paths = [dataset_paths[tuning_index]]

        date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

        if csv_filename is None:
            csv_file = os.path.join(log_dir, name + "_stats" + date_ + ".csv")
        else:
            csv_file = os.path.join(log_dir, csv_filename)

        for indexes in splits[self.config.CONFIG_CV[CVK.FIRST_SPLIT]:]:
            train_step_name = "step"
            if len(indexes) > 1:
                for i in indexes:
                    train_step_name += "_" + str(i)
            else:
                train_step_name += "_" + str(indexes[0]) + "_" + self.data_storage.get_name(np.array(paths)[indexes][0])

            leave_out_paths = np.array(paths)[indexes]

            if self.__check_data_label__(leave_out_paths):
                print(f"The patient file(s) '{', '.join(leave_out_paths)}' are no needed labels for training! "
                      f"So we skip this patient(s)!")
                continue

            cv_step_names.set_names(leave_out_names=[self.data_storage.get_name(p) for p in leave_out_paths],
                                    train_step_name=train_step_name)
            self.cross_validation_step(trainer=trainer,
                                       dataset_paths=dataset_paths,
                                       train_step_name=train_step_name,
                                       cv_step_names=cv_step_names)

            for i, path_ in enumerate(leave_out_paths):
                sensitivity, specificity = 0, 0
                with open(csv_file, 'a', newline='') as csvfile:  # for full cross_valid and for separate file
                    fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'index': str(i),
                                     'sensitivity': str(sensitivity),
                                     'specificity': str(specificity),
                                     'name': path_,
                                     'model_name': os.path.join(log_dir, train_step_name)})

    def _get_paths_and_splits(self, root_path=None):
        if root_path is None:
            root_path = self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH]
        paths = self.data_storage.get_paths(storage_path=root_path)
        number = CVK.NUMBER_SORT in self.config.CONFIG_CV.keys()
        number_sort = self.config.CONFIG_CV[CVK.NUMBER_SORT] if number else None
        paths = get_sort(paths=paths, number=number, split=number_sort)

        splits = get_splits(typ=self.config.CONFIG_CV[CVK.SPLIT_PATHS_BY], paths=paths,
                            values=self.config.CONFIG_CV[CVK.PATIENTS_EXCLUDE_FOR_TEST])

        return paths, splits

    def get_nearest_int_delimiter(self, folder):
        checkpoints_folders = glob(os.path.join(folder, 'cp-*'))
        checkpoints_folders = sorted(checkpoints_folders)

        return int(checkpoints_folders[0].split(self.config.CONFIG_PATHS[PK.SYS_DELIMITER])[-1].split('-')[-1])

    def __check_data_label__(self, paths) -> bool:
        label_not_to_train = True
        for path in paths:
            label_not_to_train = label_not_to_train & self.__check_label__(path)

        return label_not_to_train

    def __check_label__(self, path: str) -> bool:
        data = self.data_storage.get_datas(data_path=path)
        unique_y = np.unique(data[self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1]])
        intersect = np.intersect1d(unique_y, self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN])

        return True if intersect.__len__() == 0 else False

    @staticmethod
    def get_csv(search_folder):
        csv_paths = glob(os.path.join(search_folder, '*.csv'))
        if len(csv_paths) > 1:
            raise ValueError(search_folder + ' has more then one .csv files!')
        if len(csv_paths) == 0:
            raise ValueError('No .csv files were found in ' + search_folder)
        csv_path = csv_paths[0]

        return csv_path

    @staticmethod
    def get_history(search_folder):
        history_paths = utils.glob_multiple_file_types(search_folder, '.*.npy', '*.npy')
        if len(history_paths) == 0:
            print('Error! No history files were found!')
            # raise ValueError('Error! No history files were found!')
            return {}, search_folder
        if len(history_paths) > 1:
            print(f'Error! Too many history.npy files were found in {search_folder}!')
            return {}, search_folder
            # raise ValueError(f'Error! Too many .npy files were found in {model_path}!')

        history_path = history_paths[0]
        history = np.load(history_path, allow_pickle=True)

        if len(history.shape) == 0:
            history = history.item()
        return history, history_path
