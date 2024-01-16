import sys
import inspect
import os
from glob import glob
import psutil
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import provider
from configuration.copy_py_files import copy_files
from data_utils.shuffle import Shuffle
from data_utils.weights import Weights

'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.dataloader = provider.get_data_loader(typ=self.config.CONFIG_DATALOADER["TYPE"], config=self.config,
                                                   dict_names=[self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][0],
                                                               self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][1],
                                                               self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][4]])
        self.valid_archives_saving_path = None
        self.archives_of_batch_size_saving_path = None
        self.batch_size = None
        self.load_name_for_name = self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][2]
        self.load_name_for_X = self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][0]
        self.load_name_for_y = self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][1]
        self.dict_names = self.config.CONFIG_PREPROCESSOR["DICT_NAMES"]
        self.piles_number = self.config.CONFIG_PREPROCESSOR["PILES_NUMBER"]
        self.weights_filename = self.config.CONFIG_PREPROCESSOR["WEIGHT_FILENAME"]

        self.Weights = Weights(self.config, self.dataloader, self.weights_filename)

    @staticmethod
    def get_execution_flags_for_pipeline_with_all_true():
        return {
            "load_data_with_dataloader": True,
            "add_sample_weights": True,
            "scale": True,
            "shuffle": True
        }

    def pipeline(self, root_path=None, preprocessed_path=None, execution_flags=None):
        dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        print("Time before the start of preprocessing", dt_string)

        process = psutil.Process(os.getpid())

        if execution_flags is None:
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()

        if root_path is None:
            root_path = self.config.CONFIG_PATHS["RAW_SOURCE_PATH"]
        if preprocessed_path is None:
            preprocessed_path = self.config.CONFIG_PATHS["RAW_NPZ_PATH"]

        if not os.path.exists(preprocessed_path):
            os.makedirs(preprocessed_path)

        print('ROOT PATH', root_path)
        print('PREPROCESSED PATH', preprocessed_path)

        copy_files(preprocessed_path,
                   self.config.CONFIG_PREPROCESSOR["FILES_TO_COPY"],
                   self.config.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])

        print('-------------------------------------------------Memory, preprocessor 0, before preprocessing',
              process.memory_info().rss)

        # ---------Data reading part--------------
        if execution_flags['load_data_with_dataloader']:
            self.dataloader.files_read_and_save_to_npz(root_path, preprocessed_path)

        print('-------------------------------------------------Memory, preprocessor 1, after reading of origin files',
              process.memory_info().rss)

        # ----------weights part------------------
        if execution_flags['add_sample_weights']:
            weights = self.Weights.weights_get_or_save(preprocessed_path)
            self.Weights.weightedData_save(preprocessed_path, weights)

        print('-------------------------------------------------Memory, preprocessor 2, after sample weights',
              process.memory_info().rss)

        # ----------scaler part ------------------
        if execution_flags['scale'] and self.config.CONFIG_PREPROCESSOR["NORMALIZATION_TYPE"] is not None:
            print('SCALER TYPE', self.config.CONFIG_PREPROCESSOR["NORMALIZATION_TYPE"])
            scaler = provider.get_scaler(config=self.config,
                                         typ=self.config.CONFIG_PREPROCESSOR["NORMALIZATION_TYPE"],
                                         preprocessed_path=preprocessed_path,
                                         scaler_file=self.config.CONFIG_PREPROCESSOR["SCALER_FILE"],
                                         scaler_path=self.config.CONFIG_PREPROCESSOR["SCALER_PATH"],
                                         dict_names=[self.config.CONFIG_PREPROCESSOR["DICT_NAMES"][x] for x in
                                                     [0, 1, 4]])
            scaler.iterate_over_archives_and_save_scaled_X(preprocessed_path, preprocessed_path)

        print('-------------------------------------------------Memory, preprocessor 3, after scaling',
              process.memory_info().rss)

        # ----------shuffle part------------------
        if execution_flags['shuffle']:
            paths = glob(os.path.join(preprocessed_path, '*.npz'))
            shuffle = Shuffle(self.config,
                              raw_paths=paths,
                              dict_names=self.dict_names,
                              augmented=self.config.CONFIG_AUG["enable"])
            shuffle.shuffle()

        print('-------------------------------------------------Memory, preprocessor 4, after shuffling',
              process.memory_info().rss)


if __name__ == '__main__':
    from configuration.get_config import telegram, CONFIG_PREPROCESSOR
    import configuration.get_config as configuration

    execution_flags_ = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
    execution_flags_['load_data_with_dataloader'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["LOAD_DATA_WITH_DATALOADER"]
    execution_flags_['add_sample_weights'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["ADD_SAMPLE_WEIGHTS"]
    execution_flags_['scale'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["SCALE"]
    execution_flags_['shuffle'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["SHUFFLE"]

    try:
        preprocessor = Preprocessor(configuration)
        preprocessor.pipeline(execution_flags=execution_flags_)

        telegram.send_tg_message("Operations in preprocessor.py are successfully completed!")

    except Exception as e:
        telegram.send_tg_message(f"ERROR! in Preprocessor {e}")

        raise e
