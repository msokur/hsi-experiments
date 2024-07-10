import inspect
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import provider
from utils import get_used_memory
from configuration.copy_py_files import copy_files
from data_utils.weights import Weights

from configuration.keys import DataLoaderKeys as DLK, PathKeys as PK, PreprocessorKeys as PPK
from configuration.parameter import (
    STORAGE_TYPE, DATASET_TYPE, SMALL_SHUFFLE_SET
)

'''
Preprocessor contains opportunity of
1. Two step shuffling for big datasets
Link: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
2. Saving of big dataset into numpy archives of a certain(batch_size) size 
'''


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.valid_archives_saving_path = None
        self.archives_of_batch_size_saving_path = None
        self.batch_size = None
        self.load_name_for_name = self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][2]
        self.load_name_for_X = self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][0]
        self.load_name_for_y = self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][1]
        self.dict_names = self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES]
        self.piles_number = self.config.CONFIG_PREPROCESSOR[PPK.PILES_NUMBER]
        self.weights_filename = self.config.CONFIG_PREPROCESSOR[PPK.WEIGHT_FILENAME]

    @staticmethod
    def get_execution_flags_for_pipeline_with_all_true():
        return {
            PPK.EF_LOAD_DATA_WITH_DATALOADER: True,
            PPK.EF_ADD_SAMPLE_WEIGHTS: True,
            PPK.EF_SCALE: True,
            PPK.EF_SHUFFLE: True
        }

    def pipeline(self, root_path=None, preprocessed_path=None, execution_flags=None):
        dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        print("Time before the start of preprocessing", dt_string)

        process_id = os.getpid()

        if execution_flags is None:
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()

        if root_path is None:
            root_path = self.config.CONFIG_PATHS[PK.RAW_SOURCE_PATH]
        if preprocessed_path is None:
            preprocessed_path = self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH]

        if not os.path.exists(preprocessed_path):
            os.makedirs(preprocessed_path)

        data_storage = provider.get_data_storage(typ=STORAGE_TYPE)

        print('ROOT PATH', root_path)
        print('PREPROCESSED PATH', preprocessed_path)

        copy_files(preprocessed_path,
                   self.config.CONFIG_PREPROCESSOR["FILES_TO_COPY"])

        # ---------Data reading part--------------
        if execution_flags[PPK.EF_LOAD_DATA_WITH_DATALOADER]:
            cube_loader = provider.get_cube_loader(typ=self.config[DLK.FILE_EXTENSION],
                                                   config=self.config)
            mask_loader = provider.get_annotation_mask_loader(typ=self.config[DLK.MASK_EXTENSION],
                                                              config=self.config)
            dataloader = provider.get_data_loader(typ=self.config.CONFIG_DATALOADER[DLK.TYPE],
                                                  config=self.config,
                                                  cube_loader=cube_loader,
                                                  mask_loader=mask_loader,
                                                  data_storage=data_storage)
            dataloader.read_files_and_save_to_archive(root_path, preprocessed_path)

        print(f'---- Memory, preprocessor 1, after reading of origin files '
              f'{get_used_memory(process_id=process_id)} ----')

        # ----------weights part------------------
        if execution_flags[PPK.EF_ADD_SAMPLE_WEIGHTS]:
            weight_calc = Weights(filename=self.weights_filename,
                                  data_storage=data_storage,
                                  label_file=os.path.join(preprocessed_path,
                                                          self.config.CONFIG_DATALOADER[DLK.LABELS_FILENAME]),
                                  y_dict_name=self.load_name_for_y,
                                  weight_dict_name=self.dict_names[-1])
            weights = weight_calc.weights_get_or_save(preprocessed_path)
            weight_calc.weighted_data_save(preprocessed_path, weights)

        print(f'---- Memory, preprocessor 2, after sample weights {get_used_memory(process_id=process_id)} ----')

        # ----------scaler part ------------------
        if execution_flags[PPK.EF_SCALE] and self.config.CONFIG_PREPROCESSOR[PPK.NORMALIZATION_TYPE] is not None:
            print('SCALER TYPE', self.config.CONFIG_PREPROCESSOR[PPK.NORMALIZATION_TYPE])
            scaler = provider.get_scaler(config=self.config,
                                         typ=self.config.CONFIG_PREPROCESSOR[PPK.NORMALIZATION_TYPE],
                                         data_storage=data_storage,
                                         preprocessed_path=preprocessed_path,
                                         scaler_file=self.config.CONFIG_PREPROCESSOR[PPK.SCALER_FILE],
                                         scaler_path=self.config.CONFIG_PREPROCESSOR[PPK.SCALER_PATH],
                                         dict_names=[self.config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES][x] for x in
                                                     [0, 1, 4]])
            scaler.iterate_over_archives_and_save_scaled_X(destination_path=preprocessed_path)

        print(f'---- Memory, preprocessor 3, after scaling {get_used_memory(process_id=process_id)}----')

        # ----------shuffle part------------------
        if execution_flags[PPK.EF_SHUFFLE]:
            print(f"SHUFFLE PATHS {self.config.CONFIG_PATHS[PK.SHUFFLED_PATH]}")
            shuffle = provider.get_shuffle(config=self.config, typ=DATASET_TYPE, data_storage=data_storage,
                                           raw_path=preprocessed_path,
                                           dict_names=self.dict_names, small=SMALL_SHUFFLE_SET)
            shuffle.shuffle()

        print(f'---- Memory, preprocessor 4, after shuffling {get_used_memory(process_id=process_id)} ----')


if __name__ == '__main__':
    from configuration.get_config import Config

    try:
        preprocessor = Preprocessor(config=Config)
        preprocessor.pipeline(execution_flags=Config.CONFIG_PREPROCESSOR[PPK.EXECUTION_FLAGS])

        Config.telegram.send_tg_message("Operations in preprocessor.py are successfully completed!")

    except Exception as e:
        Config.telegram.send_tg_message(f"ERROR! in Preprocessor {e}")

        raise e
