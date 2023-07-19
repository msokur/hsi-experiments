import sys
import inspect
import os
import numpy as np
import glob
from tqdm import tqdm
import pickle

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import provider
from configuration.copy_py_files import copy_files
from data_utils.shuffle import Shuffle

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
        self.dataloader = provider.get_data_loader(typ=self.CONFIG_DATALOADER["TYPE"],
                                                   dict_names=[self.CONFIG_PREPROCESSOR["DICT_NAMES"][0],
                                                               self.CONFIG_PREPROCESSOR["DICT_NAMES"][1],
                                                               self.CONFIG_PREPROCESSOR["DICT_NAMES"][4]])
        self.valid_archives_saving_path = None
        self.archives_of_batch_size_saving_path = None
        self.batch_size = None
        self.load_name_for_name = self.CONFIG_PREPROCESSOR["DICT_NAMES"][2]
        self.load_name_for_X = self.CONFIG_PREPROCESSOR["DICT_NAMES"][0]
        self.load_name_for_y = self.CONFIG_PREPROCESSOR["DICT_NAMES"][1]
        self.dict_names = self.CONFIG_PREPROCESSOR["DICT_NAMES"]
        self.piles_number = self.CONFIG_PREPROCESSOR["PILES_NUMBER"]
        self.weights_filename = self.CONFIG_PREPROCESSOR["WEIGHT_FILENAME"]

    def weights_get_from_file(self, root_path):
        weights_path = os.path.join(root_path, self.weights_filename)
        if os.path.isfile(weights_path):
            weights = pickle.load(open(weights_path, 'rb'))
            return weights['weights']
        else:
            raise ValueError("No .weights file was found in the directory, check given path")

    def weights_get_or_save(self, root_path):
        weights_path = os.path.join(root_path, self.weights_filename)

        paths = glob.glob(os.path.join(root_path, '*.npz'))
        y_unique = pickle.load(open(os.path.join(root_path, self.dataloader.get_labels_filename()), 'rb'))

        quantities = []
        for path in tqdm(paths):
            data = np.load(path)
            X, y = data['X'], data['y']

            quantity = []
            for y_u in y_unique:
                quantity.append(X[y == y_u].shape[0])

            quantities.append(quantity)

        quantities = np.array(quantities)

        sum_ = np.sum(quantities[:, self.CONFIG_DATALOADER["LABELS"]])
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = sum_ / quantities

        weights[np.isinf(weights)] = 0

        data = {
            'weights': weights,
            'sum': sum_,
            'quantities': quantities
        }

        with open(weights_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return weights

    def weightedData_save(self, root_path, weights):
        paths = glob.glob(os.path.join(root_path, "*.npz"))
        for i, path in tqdm(enumerate(paths)):
            data = np.load(path)
            X, y = data["X"], data["y"]
            weights_ = np.zeros(y.shape)

            for j in np.unique(y):
                weights_[y == j] = weights[i, j]

            data_ = {n: a for n, a in data.items()}
            data_["weights"] = weights_

            np.savez(os.path.join(root_path, self.dataloader.get_name(path)), **data_)

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
            root_path = self.CONFIG_PATHS["RAW_SOURCE_PATH"]
        if preprocessed_path is None:
            preprocessed_path = self.CONFIG_PATHS["RAW_NPZ_PATH"]

        if not os.path.exists(preprocessed_path):
            os.makedirs(preprocessed_path)

        print('ROOT PATH', root_path)
        print('PREPROCESSED PATH', preprocessed_path)

        copy_files(preprocessed_path,
                   self.CONFIG_PREPROCESSOR["FILES_TO_COPY"],
                   self.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])

        # ---------Data reading part--------------
        if execution_flags['load_data_with_dataloader']:
            self.dataloader.files_read_and_save_to_npz(root_path, preprocessed_path)

        # ----------weights part------------------
        if execution_flags['add_sample_weights']:
            weights = self.weights_get_or_save(preprocessed_path)
            self.weightedData_save(preprocessed_path, weights)

        # ----------scaler part ------------------
        if execution_flags['scale'] and self.CONFIG_PREPROCESSOR["NORMALIZATION_TYPE"] is not None:
            print('SCALER TYPE', self.CONFIG_PREPROCESSOR["NORMALIZATION_TYPE"])
            scaler = provider.get_scaler(typ=self.CONFIG_PREPROCESSOR["NORMALIZATION_TYPE"],
                                         preprocessed_path=preprocessed_path,
                                         scaler_file=self.CONFIG_PREPROCESSOR["SCALER_FILE"],
                                         scaler_path=self.CONFIG_PREPROCESSOR["SCALER_PATH"],
                                         dict_names=[self.CONFIG_PREPROCESSOR["DICT_NAMES"][x] for x in [0, 1, 4]])
            scaler.iterate_over_archives_and_save_scaled_X(preprocessed_path, preprocessed_path)

        # ----------shuffle part------------------
        if execution_flags['shuffle']:
            paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
            shuffle = Shuffle(raw_paths=paths,
                              dict_names=self.dict_names,
                              preprocessor_conf=self.CONFIG_PREPROCESSOR,
                              paths_conf=self.CONFIG_PATHS,
                              augmented=self.CONFIG_AUG["enable"])
            shuffle.shuffle()


if __name__ == '__main__':
    from configuration.get_config import telegram, CONFIG_PATHS, CONFIG_PREPROCESSOR, CONFIG_DATALOADER, CONFIG_AUG

    execution_flags_ = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
    execution_flags_['load_data_with_dataloader'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["LOAD_DATA_WITH_DATALOADER"]
    execution_flags_['add_sample_weights'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["ADD_SAMPLE_WEIGHTS"]
    execution_flags_['scale'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["SCALE"]
    execution_flags_['shuffle'] = CONFIG_PREPROCESSOR["EXECUTION_FLAGS"]["SHUFFLE"]

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
