import os
from glob import glob
import datetime
from typing import List

import numpy as np
import csv

import utils
import provider

from data_utils.dataset.choice_names import ChoiceNames

from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK, DataLoaderKeys as DLK, \
    PreprocessorKeys as PPK
from configuration.parameter import (
    ARCHIVE_TYPE,
)


class CrossValidatorBase:
    def __init__(self, config):
        self.config = config
        self.data_archive = provider.get_data_archive(typ=ARCHIVE_TYPE)

    @staticmethod
    def get_execution_flags():
        return {
            CVK.EF_CROSS_VALIDATION: True,
            CVK.EF_EVALUATION: True
        }

    def pipeline(self, execution_flags=None, **kwargs):
        if execution_flags is None:
            execution_flags = CrossValidatorBase.get_execution_flags()

        if execution_flags[CVK.EF_CROSS_VALIDATION]:
            self.cross_validation()
        if execution_flags[CVK.EF_EVALUATION]:
            self.evaluation(**kwargs)

        self.config.telegram.send_tg_message(f'Operations in cross_validation.py for {self.config.CONFIG_CV["NAME"]} '
                                             f'are successfully completed!')

    def evaluation(self, **kwargs):
        training_csv_path = self.get_csv(os.path.join(self.config.CONFIG_PATHS[PK.LOGS_FOLDER][0],
                                                      self.config.CONFIG_CV[CVK.NAME]))
        print('training_csv_path', training_csv_path)

        evaluator = provider.get_evaluation(config=self.config,
                                            labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN])

        evaluator.save_predictions_and_metrics(training_csv_path=training_csv_path,
                                               data_folder=self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
                                               **kwargs)

    def cross_validation_step(self, model_name: str, except_names: List[str], except_cv_names=None):
        if except_cv_names is None:
            except_cv_names = []
        choice_names = ChoiceNames(data_archive=self.data_archive, config_cv=self.config.CONFIG_CV,
                                   labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                   y_dict_name=self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1],
                                   log_dir=model_name)
        except_valid_names = choice_names.get_valid_except_names(raw_path=self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
                                                                 except_names=except_cv_names)
        except_train_names = list(set(except_names) - set(except_cv_names) - set(except_valid_names))

        print(f"We except for patient out data: {', '.join(n for n in except_cv_names)}.\n")
        print(f"We except for train data: {', '.join(n for n in except_train_names)}.\n")
        print(f"We except for valid data: {', '.join(n for n in except_valid_names)}.\n")

        trainer = provider.get_trainer(config=self.config, data_archive=self.data_archive,
                                       typ=self.config.CONFIG_TRAINER["TYPE"],
                                       model_name=model_name,
                                       except_indexes=except_names)
        trainer.train()

    """def get_paths_and_splits(self, root_path=None):
        if root_path is None:
            root_path = self.config.CONFIG_PATHS["RAW_NPZ_PATH"]

        paths = glob(os.path.join(root_path, "*.npz"))
        extension = provider.get_extension_loader(config=self.config,
                                                  typ=self.config.CONFIG_DATALOADER["FILE_EXTENSION"])
        paths = extension.sort(paths)

        splits = get_splits(typ=self.config.CONFIG_DATALOADER["SPLIT_PATHS_BY"], paths=paths,
                            values=self.config.CONFIG_CV["HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST"],
                            delimiter=self.config.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])

        return paths, splits"""

    def cross_validation(self, csv_filename=None):
        name = self.config.CONFIG_CV[CVK.NAME]
        self.config.CONFIG_PATHS[PK.LOGS_FOLDER].append(name)

        root_folder = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER])
        path_template = os.path.join(*self.config.CONFIG_PATHS[PK.LOGS_FOLDER], 'step')

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        data_loader = provider.get_data_loader(config=self.config, typ=self.config.CONFIG_DATALOADER[DLK.TYPE],
                                               data_archive=self.data_archive)
        paths, splits = data_loader.get_paths_and_splits()

        date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

        if csv_filename is None:
            csv_filename = os.path.join(root_folder, name + "_stats" + date_ + ".csv")

        for indexes in splits[self.config.CONFIG_CV[CVK.FIRST_SPLIT]:]:
            model_name = path_template
            if len(indexes) > 1:
                for i in indexes:
                    model_name += "_" + str(i)
            else:
                model_name += "_" + str(indexes[0]) + "_" + data_loader.get_name(np.array(paths)[indexes][0])

            paths_patch = np.array(paths)[indexes]

            if self.__check_data_label__(paths_patch):
                print(f"In files {paths_patch} are no needed labels for training!")
                continue

            except_names = [data_loader.get_name(path=p) for p in paths]
            self.cross_validation_step(model_name=model_name, except_names=except_names,
                                       except_cv_names=[data_loader.get_name(p) for p in paths_patch])

            for i, path_ in enumerate(paths_patch):
                sensitivity, specificity = 0, 0
                with open(csv_filename, 'a', newline='') as csvfile:  # for full cross_valid and for separate file
                    fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'index': str(i),
                                     'sensitivity': str(sensitivity),
                                     'specificity': str(specificity),
                                     'name': path_,
                                     # 'model_name': config.MODEL_NAME})
                                     'model_name': model_name})

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
        data = self.data_archive.get_datas(data_path=path)
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
