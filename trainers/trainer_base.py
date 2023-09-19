from typing import List

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import abc
import pickle

from data_utils.batches import NameBatchSplit, DataGenerator
from util.compare_distributions import DistributionsChecker
from data_utils.weights import Weights

from data_utils.data_archive import DataArchive
from configuration.copy_py_files import copy_files
from configuration.get_config import telegram
from configuration.keys import TrainerKeys as TK, PathKeys as PK
from configuration.parameter import (
    VALID_LOG, HISTORY_FILE, TUNE, TRAIN, VALID
)


class Trainer:
    def __init__(self, data_archive: DataArchive, config_trainer: dict, config_paths: dict, labels_to_train: List[int],
                 model_name: str, except_cv_names: List[str], except_train_names: List[str],
                 except_valid_names: List[str], dict_names: List[str], config_distribution: dict):
        self.data_archive = data_archive
        self.CONFIG_TRAINER = config_trainer
        self.CONFIG_PATHS = config_paths
        self.labels_to_train = labels_to_train
        self.log_dir = model_name
        self.except_cv_names = except_cv_names
        self.except_train_names = except_train_names
        self.except_valid_names = except_valid_names
        self.dict_names = dict_names
        self.CONFIG_DISTRIBUTION = config_distribution
        self.batch_path = None
        self.mirrored_strategy = None

    @abc.abstractmethod
    def train_process(self):
        pass

    def save_except_names(self, except_names):
        with open(os.path.join(self.log_dir, VALID_LOG), "wb") as f:
            pickle.dump(except_names, f, pickle.HIGHEST_PROTOCOL)

    def logging_and_copying(self):
        if not self.CONFIG_TRAINER[TK.RESTORE]:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            copy_files(self.log_dir, self.CONFIG_TRAINER["FILES_TO_COPY"])

    def get_datasets(self, for_tuning=False):
        root_data_paths = self.data_archive.get_paths(archive_path=self.CONFIG_PATHS[PK.SHUFFLED_PATH])
        self.batch_path = self.CONFIG_PATHS[PK.BATCHED_PATH]
        if len(self.except_cv_names) > 0:
            self.batch_path += "_" + self.except_cv_names[0]
            if for_tuning:
                self.batch_path += "_" + TUNE
                ds = DistributionsChecker(data_archive=self.data_archive, path=os.path.split(root_data_paths[0])[0],
                                          config_distribution=self.CONFIG_DISTRIBUTION,
                                          check_dict_name=self.dict_names[0])
                tuning_index = ds.get_small_database_for_tuning()
                root_data_paths = [root_data_paths[tuning_index]]

        if not os.path.exists(path=self.batch_path):
            os.mkdir(path=self.batch_path)

        batch_split = NameBatchSplit(data_archive=self.data_archive, batch_size=self.CONFIG_TRAINER[TK.BATCH_SIZE],
                                     use_labels=self.labels_to_train, dict_names=self.dict_names,
                                     with_sample_weights=self.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS])
        train_paths = batch_split.split(data_paths=root_data_paths,
                                        batch_save_path=os.path.join(self.batch_path, TRAIN),
                                        except_names=self.except_train_names)
        valid_paths = batch_split.split(data_paths=root_data_paths,
                                        batch_save_path=os.path.join(self.batch_path, VALID),
                                        except_names=self.except_valid_names)
        self.save_except_names(except_names=self.except_valid_names)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_ds = self.__get_dataset__(batch_paths=train_paths, options=options)
        valid_ds = self.__get_dataset__(batch_paths=valid_paths, options=options)

        weights = Weights(filename="", data_archive=self.data_archive, labels=self.labels_to_train,
                          y_dict_name=self.dict_names[1])

        class_weights = weights.get_class_weights(class_data_paths=train_paths)
        print(class_weights)
        # TODO class_weights dirty fix
        class_weights = {k: v for k, v in enumerate(class_weights.values())}
        return train_ds, valid_ds, class_weights

    def get_callbacks(self):
        checkpoint_path = os.path.join(self.log_dir, self.CONFIG_PATHS[PK.CHECKPOINT_PATH], "cp-{epoch:04d}")

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.CONFIG_TRAINER[TK.MODEL_CHECKPOINT]["monitor"],
            verbose=1,
            save_best_only=self.CONFIG_TRAINER[TK.MODEL_CHECKPOINT]["save_best_only"],
            mode=self.CONFIG_TRAINER[TK.MODEL_CHECKPOINT]["mode"])

        callbacks_ = [checkpoints_callback]
        if self.CONFIG_TRAINER[TK.EARLY_STOPPING]["enable"]:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=self.CONFIG_TRAINER[TK.EARLY_STOPPING]["monitor"],
                mode=self.CONFIG_TRAINER[TK.EARLY_STOPPING]["mode"],
                min_delta=self.CONFIG_TRAINER[TK.EARLY_STOPPING]["min_delta"],
                patience=self.CONFIG_TRAINER[TK.EARLY_STOPPING]["patience"],
                verbose=1,
                restore_best_weights=self.CONFIG_TRAINER[TK.EARLY_STOPPING]["restore_best_weights"])

            callbacks_.append(early_stopping_callback)

        return callbacks_

    def save_history(self, history):
        np.save(os.path.join(self.log_dir, HISTORY_FILE), history.history)

    def train(self):
        try:
            if self.CONFIG_PATHS[PK.MODE] == "WITH_GPU":
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                    except RuntimeError as e:
                        print(e)

                self.mirrored_strategy = tf.distribute.experimental.CentralStorageStrategy()
                # self.mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
            elif self.CONFIG_PATHS[PK.MODE] != "WITHOUT_GPU":
                print(f"ERROR Mode: {self.CONFIG_PATHS[PK.MODE]} not available! Continue without GPU strategy")
        except Exception as e:
            if telegram is not None:
                telegram.send_tg_message(f'ERROR!!!, training {self.log_dir} has finished with error {e}')
            raise e  # TODO REMOVE!!

        model, history = self.train_process()

        checkpoints_paths = os.path.join(self.log_dir, self.CONFIG_PATHS[PK.CHECKPOINT_PATH])
        if not os.path.exists(checkpoints_paths):
            os.mkdir(checkpoints_paths)

        if not self.CONFIG_TRAINER[TK.EARLY_STOPPING]["enable"]:
            final_model_save_path = os.path.join(checkpoints_paths, f'cp-{len(history.history["loss"]):04d}')
            if not os.path.exists(final_model_save_path):
                os.mkdir(final_model_save_path)
            model.save(final_model_save_path)

        return model, history

    def get_output_shape(self):
        data_paths = self.data_archive.get_paths(archive_path=self.CONFIG_PATHS[PK.SHUFFLED_PATH])
        X = self.data_archive.get_data(data_path=data_paths[0], data_name=self.dict_names[0])

        return X.shape[1:]

    def __get_dataset__(self, batch_paths: List[str], options: tf.data.Options):
        dataset = DataGenerator(data_archive=self.data_archive, batch_paths=batch_paths, X_name=self.dict_names[0],
                                y_name=self.dict_names[1], weights_name=self.dict_names[5],
                                with_sample_weights=self.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS])
        tf_dataset = tf.data.Dataset.from_generator(generator=dataset, output_signature=dataset.get_output_signature())
        return tf_dataset.with_options(options=options)

    @staticmethod
    def __set_tf_option__():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


if __name__ == '__main__':
    pass
