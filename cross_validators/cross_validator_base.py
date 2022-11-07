import os
from glob import glob
import abc
import datetime
import numpy as np
import csv
import inspect

import config
import utils
from provider import get_trainer, get_data_loader
from data_utils.data_loaders.data_loader_base import DataLoader


class CrossValidatorBase:
    def __init__(self, name):
        self.name = name

        current_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        project_folder = os.path.dirname(current_folder)
        self.project_folder = project_folder

    @staticmethod
    def get_execution_flags():
        return {
            "cross_validation": True,
            "evaluation": True
        }

    def pipeline(self, execution_flags=None, **kwargs):
        if execution_flags is None:
            execution_flags = CrossValidatorBase.get_execution_flags()

        if execution_flags['cross_validation']:
            CrossValidatorBase.cross_validation(self.name)
        if execution_flags['evaluation']:
            self.evaluation(**kwargs)

        utils.send_tg_message(
            f'{config.USER}, operations in cross_validation.py for {self.name} are successfully completed!')

    @abc.abstractmethod
    def evaluation(self, **kwargs):  # has to be implemented in child classes
        pass

    @staticmethod
    def cross_validation_step(model_name, except_names=None):
        if except_names is None:
            except_names = []
        trainer = get_trainer(model_name=model_name, except_indexes=except_names)
        trainer.train()

    @staticmethod
    def cross_validation(root_folder_name, csv_filename=None):
        config.MODEL_NAME_PATHS.append(root_folder_name)

        root_folder = os.path.join(*config.MODEL_NAME_PATHS)
        config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)

        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        data_loader = get_data_loader()
        paths, splits = data_loader.get_paths_and_splits()

        date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

        if csv_filename is None:
            csv_filename = os.path.join(root_folder, root_folder_name + '_stats' + date_ + '.csv')

        for indexes in splits[config.CV_FIRST_SPLIT:]:
            model_name = config.MODEL_NAME  # config.MODEL_NAME
            if len(indexes) > 1:
                for i in indexes:
                    model_name += '_' + str(i)
            else:
                model_name += '_' + str(indexes[0]) + '_' + DataLoader.get_name_easy(np.array(paths)[indexes][0])

            paths_patch = np.array(paths)[indexes]

            CrossValidatorBase.cross_validation_step(model_name,
                                                     except_names=[DataLoader.get_name_easy(p) for p in paths_patch])

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

    @staticmethod
    def get_nearest_int_delimiter(folder):
        checkpoints_folders = glob(os.path.join(folder, 'cp-*'))
        checkpoints_folders = sorted(checkpoints_folders)

        return int(checkpoints_folders[0].split(config.SYSTEM_PATHS_DELIMITER)[-1].split('-')[-1])

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
        # print(history_paths)
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
        # print(history)
        if len(history.shape) == 0:
            history = history.item()
        return history, history_path
