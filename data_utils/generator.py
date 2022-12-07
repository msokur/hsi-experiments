import os
import glob
import numpy as np
from numpy import ndarray
from tensorflow import keras
import pickle

from data_utils.batch_split import BatchSplit
from configuration.get_config import PATHS, CV, TRAINER, DATALOADER, PREPRO
from util import compare_distributions


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    # modes: 'train', 'valid', 'all'
    def __init__(self,
                 mode: str,
                 shuffled_npz_path: str,
                 batches_npz_path: str,
                 batch_size: int,
                 split_factor: float,
                 except_indexes=None,
                 valid_except_indexes=None,
                 split_flag=True,
                 for_tuning=False,
                 log_dir=None):
        if valid_except_indexes is None:
            valid_except_indexes = []
        if except_indexes is None:
            except_indexes = []
        self.class_weight = None

        """Initialization"""
        self.raw_npz_path = os.path.dirname(shuffled_npz_path)
        self.shuffled_npz_path = shuffled_npz_path
        self.batches_npz_path = batches_npz_path
        self.mode = mode
        self.split_factor = split_factor
        self.split_flag = split_flag
        self.for_tuning = for_tuning
        self.log_dir = log_dir

        self.shuffled_npz_paths = glob.glob(os.path.join(shuffled_npz_path, "shuffl*.npz"))

        self.batch_size = batch_size
        self.except_indexes = except_indexes
        self.valid_except_indexes = valid_except_indexes
        self.valid_except_indexes = self.get_valid_except_names()
        self.index = 0

        self.batch_split = BatchSplit(labels_to_train=DATALOADER["LABELS_TO_TRAIN"], dict_names=PREPRO["DICT_NAMES"],
                                      batch_size=self.batch_size)

        print("--------------------PARAMS----------------------")
        print(", \n".join("%s: %s" % item for item in vars(self).items()))
        print("------------------------------------------------")

        self.split()
        self.len = self.__len__()
        print("self.len", self.len)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.batches_npz_path)

    def __getitem__(self, index):
        """Generate one batch of data"""
        data = np.load(self.batches_npz_path[index])
        X, y = data["X"], data["y"]

        if TRAINER["WITH_SAMPLE_WEIGHTS"] and "weights" in data.keys():
            return X, y, data["weights"]

        return X, y.astype(np.float)

    ''' #Self-test public method - the copy of private __getitem__'''

    def getitem(self, index):
        return self.__getitem__(index)

    def split(self, except_indexes=None):
        if except_indexes is not None:
            self.except_indexes = except_indexes

        if self.for_tuning and self.split_flag:
            ds = compare_distributions.DistributionsChecker(self.shuffled_npz_path)
            tuning_index = ds.get_small_database_for_tuning()
            self.shuffled_npz_paths = [self.shuffled_npz_paths[tuning_index]]

        if self.split_flag:
            self.batch_split.split_data_into_npz_of_batch_size(self.shuffled_npz_paths,
                                                               self.batches_npz_path,
                                                               except_names=self.except_indexes,
                                                               valid_except_names=self.valid_except_indexes)
        else:
            print("!!!!!   Dataset is not split   !!!!!")

        batches_paths = glob.glob(os.path.join(self.batches_npz_path, "*.npz"))
        valid_batches_paths = glob.glob(os.path.join(self.batches_npz_path,
                                                     "valid", "*.npz"))

        if self.mode == "all":
            self.batches_npz_path = batches_paths + valid_batches_paths  # batches_paths.copy()
        if self.mode == "train":
            self.batches_npz_path = batches_paths  # [:split_factor]
        if self.mode == "valid":
            self.batches_npz_path = valid_batches_paths  # batches_paths[split_factor:]

    def get_valid_except_names(self):
        if len(self.valid_except_indexes) == 0:
            if CV["CHOOSE_EXCLUDED_VALID"] == "restore":
                print("Restore names of patients that will be used for validation dataset")
                restore_paths = glob.glob(os.path.join(CV["RESTORE_VALID_PATH"], f"*{PATHS['SYSTEM_PATHS_DELIMITER']}"))
                restore_path = restore_paths[np.flatnonzero(
                    np.core.defchararray.find(restore_paths, CV["RESTORE_VALID_SEQUENCE"]) != -1)[0]]

                log_name = self.log_dir.split(PATHS["SYSTEM_PATHS_DELIMITER"])[-1]
                log_index = log_name.split("_")[1]  # can be problems

                restore_log_paths = glob.glob(os.path.join(restore_path, f"*{PATHS['SYSTEM_PATHS_DELIMITER']}"))
                restore_log_path = restore_log_paths[
                    np.flatnonzero(np.core.defchararray.find(restore_log_paths, "3d_" + str(log_index) + "_") != -1)[0]]

                valid_except_indexes = pickle.load(
                    open(os.path.join(restore_log_path, "valid.valid_except_names"), "rb"))
                print(
                    f"We restore {valid_except_indexes} from {restore_log_path} with {CV['RESTORE_VALID_SEQUENCE']}")
                return valid_except_indexes

            raw_paths = glob.glob(os.path.join(self.raw_npz_path, '*.npz'))
            raw_paths_names = [r.split(PATHS["SYSTEM_PATHS_DELIMITER"])[-1].split('.')[0] for r in raw_paths]

            print('Getting new validation patients')
            if CV["CHOOSE_EXCLUDED_VALID"] == "randomly":
                return DataGenerator.get_random_choice(paths=raw_paths_names,
                                                       excepts=self.except_indexes,
                                                       size=CV["HOW_MANY_VALID_EXCLUDE"])

            elif CV["CHOOSE_EXCLUDED_VALID"] == "by_class":
                return DataGenerator.choose_path(paths=raw_paths,
                                                 paths_names=raw_paths_names,
                                                 excepts=self.except_indexes)

        print('Return existing validation patients')
        return self.valid_except_indexes

    @staticmethod
    def choose_path(paths, paths_names, excepts, classes=None) -> ndarray:
        if classes is None:
            classes = np.array([])
        valid = DataGenerator.get_random_choice(paths_names, excepts)

        path_idx = paths_names.index(valid[0])
        data = np.load(paths[path_idx])
        unique_classes = np.unique(data['y'])
        con_classes = np.concatenate((classes, unique_classes))
        con_unique_classes = np.intersect1d(con_classes, DATALOADER["LABELS_TO_TRAIN"])
        if len(con_unique_classes) >= len(DATALOADER["LABELS_TO_TRAIN"]):
            return valid
        elif len(con_unique_classes) - len(classes) >= 1:
            return np.concatenate((valid, DataGenerator.choose_path(paths,
                                                                    paths_names,
                                                                    np.concatenate((excepts, valid)),
                                                                    con_unique_classes)))
        else:
            return DataGenerator.choose_path(paths,
                                             paths_names,
                                             np.concatenate((excepts, valid)),
                                             classes)

    @staticmethod
    def get_random_choice(paths, excepts, size=1):
        return np.random.choice([r for r in paths if r not in excepts],
                                size=size,
                                replace=False)

    def get_class_weights(self, labels=None):
        if labels is None:
            labels = DATALOADER["LABELS_TO_TRAIN"]
        labels = np.array(labels)
        sums = np.zeros(labels.shape)

        for p in self.batches_npz_path:
            data = np.load(p)
            y = data['y']
            for i, l in enumerate(labels):
                sums[i] += np.flatnonzero(y == l).shape[0]

        total = np.sum(sums)

        weights = {}
        for i, l in enumerate(labels):
            with np.errstate(divide="ignore", invalid="ignore"):
                weights[l] = (1 / sums[i]) * total / len(DATALOADER["LABELS_TO_TRAIN"])
            if weights[l] == np.inf:
                weights[l] = 0.0

        self.class_weight = weights

        return self.class_weight

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.index = 0


if __name__ == '__main__':
    pass
