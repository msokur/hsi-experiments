import sys
import inspect
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import provider
from configuration.copy_py_files import copy_files
from data_utils.weights import Weights
from data_utils.shuffle import Shuffle

from configuration.keys import DataLoaderKeys as DLK, PathKeys as PK, PreprocessorKeys as PPK, AugKeys as AK
from configuration.parameter import (
    ARCHIVE_TYPE
)

'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''


class Preprocessor:
    def __init__(self, config_preprocessor: dict, config_paths: dict, config_dataloader: dict,
                 config_augmentation: dict):
        self.CONFIG_PREPROCESSOR = config_preprocessor
        self.CONFIG_PATHS = config_paths
        self.CONFIG_DATALOADER = config_dataloader
        self.CONFIG_AUG = config_augmentation
        self.valid_archives_saving_path = None
        self.archives_of_batch_size_saving_path = None
        self.batch_size = None
        self.load_name_for_name = self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][2]
        self.load_name_for_X = self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][0]
        self.load_name_for_y = self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1]
        self.dict_names = self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES]
        self.piles_number = self.CONFIG_PREPROCESSOR[PPK.PILES_NUMBER]
        self.weights_filename = self.CONFIG_PREPROCESSOR[PPK.WEIGHT_FILENAME]

    @staticmethod
    def get_execution_flags_for_pipeline_with_all_true():
        return {
            "load_data_with_dataloader": True,
            "add_sample_weights": True,
            "scale": True,
            "shuffle": True
        }

    def pipeline(self, root_path=None, preprocessed_path=None, execution_flags=None):
        if execution_flags is None:
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()

        if root_path is None:
            root_path = self.CONFIG_PATHS[PK.RAW_NPZ_PATH]
        if preprocessed_path is None:
            preprocessed_path = self.CONFIG_PATHS[PK.RAW_NPZ_PATH]

        if not os.path.exists(preprocessed_path):
            os.makedirs(preprocessed_path)

        data_archive = provider.get_data_archive(typ=ARCHIVE_TYPE)

        print('ROOT PATH', root_path)
        print('PREPROCESSED PATH', preprocessed_path)

        copy_files(preprocessed_path,
                   self.CONFIG_PREPROCESSOR["FILES_TO_COPY"])

        # ---------Data reading part--------------
        if execution_flags['load_data_with_dataloader']:
            dataloader = provider.get_data_loader(typ=self.CONFIG_DATALOADER[DLK.TYPE],
                                                  data_archive=data_archive,
                                                  config_dataloader=self.CONFIG_DATALOADER,
                                                  config_paths=self.CONFIG_PATHS,
                                                  dict_names=[self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][0],
                                                              self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1],
                                                              self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][4]])
            dataloader.files_read_and_save_to_archive(root_path, preprocessed_path)

        # ----------weights part------------------
        if execution_flags['add_sample_weights']:
            weight_calc = Weights(filename=self.weights_filename,
                                  data_archive=data_archive,
                                  label_file=CONFIG_DATALOADER[DLK.LABELS_FILENAME],
                                  y_dict_name=self.load_name_for_y,
                                  weight_dict_name=self.dict_names[-1])
            weights = weight_calc.weights_get_or_save(preprocessed_path)
            weight_calc.weighted_data_save(preprocessed_path, weights)

        # ----------scaler part ------------------
        if execution_flags['scale'] and self.CONFIG_PREPROCESSOR[PPK.NORMALIZATION_TYPE] is not None:
            print('SCALER TYPE', self.CONFIG_PREPROCESSOR[PPK.NORMALIZATION_TYPE])
            scaler = provider.get_scaler(typ=self.CONFIG_PREPROCESSOR[PPK.NORMALIZATION_TYPE],
                                         data_archive=data_archive,
                                         preprocessed_path=preprocessed_path,
                                         scaler_file=self.CONFIG_PREPROCESSOR[PPK.SCALER_FILE],
                                         scaler_path=self.CONFIG_PREPROCESSOR[PPK.SCALER_PATH],
                                         dict_names=[self.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][x] for x in [0, 1, 4]])
            scaler.iterate_over_archives_and_save_scaled_X(preprocessed_path, preprocessed_path)

        # ----------shuffle part------------------
        if execution_flags['shuffle']:
            shuffle = Shuffle(data_archive=data_archive, raw_path=preprocessed_path, dict_names=self.dict_names,
                              piles_number=self.CONFIG_PREPROCESSOR[PPK.PILES_NUMBER],
                              shuffle_saving_path=self.CONFIG_PATHS[PK.SHUFFLED_PATH],
                              augmented=self.CONFIG_AUG[AK.ENABLE],
                              files_to_copy=self.CONFIG_PREPROCESSOR["FILES_TO_COPY"])
            shuffle.shuffle()


if __name__ == '__main__':
    from configuration.get_config import telegram, CONFIG_PATHS, CONFIG_PREPROCESSOR, CONFIG_DATALOADER, CONFIG_AUG

    execution_flags_ = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
    execution_flags_['load_data_with_dataloader'] = CONFIG_PREPROCESSOR[PPK.EXECUTION_FLAGS][PPK.EF_LOAD_DATA_WITH_DATALOADER]
    execution_flags_['add_sample_weights'] = CONFIG_PREPROCESSOR[PPK.EXECUTION_FLAGS][PPK.EF_ADD_SAMPLE_WEIGHTS]
    execution_flags_['scale'] = CONFIG_PREPROCESSOR[PPK.EXECUTION_FLAGS][PPK.EF_SCALE]
    execution_flags_['shuffle'] = CONFIG_PREPROCESSOR[PPK.EXECUTION_FLAGS][PPK.EF_SHUFFLE]

    try:
        preprocessor = Preprocessor(config_preprocessor=CONFIG_PREPROCESSOR, config_paths=CONFIG_PATHS,
                                    config_dataloader=CONFIG_DATALOADER, config_augmentation=CONFIG_AUG)
        preprocessor.pipeline(execution_flags=execution_flags_)

        if telegram is not None:
            telegram.send_tg_message("Operations in preprocessor.py are successfully completed!")

    except Exception as e:
        if telegram is not None:
            telegram.send_tg_message(f"ERROR! in Preprocessor {e}")

        raise e
