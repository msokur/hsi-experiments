from glob import glob
from typing import List, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import abc
import pickle
import psutil

from util.compare_distributions import DistributionsChecker
from data_utils.dataset.meta_files import get_cw_from_meta
from provider import get_dataset

from callbacks import CustomTensorboardCallback
from data_utils.data_archive import DataArchive
from configuration.copy_py_files import copy_files
from configuration.keys import (
    TrainerKeys as TK, PathKeys as PK, DataLoaderKeys as DLK, PreprocessorKeys as PPK,
    CrossValidationKeys as CVK)
from configuration.parameter import (
    DATASET_TYPE, VALID_LOG, HISTORY_FILE, TUNE
)


class Trainer:
    def __init__(self, config, data_archive: DataArchive, model_name: str, except_cv_names: List[str],
                 except_train_names: List[str], except_valid_names: List[str],):
        self.config = config
        self.data_archive = data_archive
        self.except_cv_names = except_cv_names
        self.except_train_names = except_train_names
        self.except_valid_names = except_valid_names
        self.batch_path = None
        self.mirrored_strategy = None
        self.log_dir = model_name
        self.dataset = get_dataset(typ=DATASET_TYPE, batch_size=config.CONFIG_TRAINER[TK.BATCH_SIZE],
                                   d3=config.CONFIG_DATALOADER[DLK.D3],
                                   with_sample_weights=config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS],
                                   data_archive=self.data_archive,
                                   dict_names=config.CONFIG_PREPROCESSOR[PPK.DICT_NAMES])

    @abc.abstractmethod
    def train_process(self):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_parameters_for_compile(self):
        # should return Loss and Metrics
        pass

    def save_except_names(self, except_names):
        with open(os.path.join(self.log_dir, VALID_LOG), "wb") as f:
            pickle.dump(except_names, f, pickle.HIGHEST_PROTOCOL)

    def logging_and_copying(self):
        if not self.config.CONFIG_TRAINER[TK.RESTORE]:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            copy_files(self.log_dir, self.config.CONFIG_TRAINER["FILES_TO_COPY"])

    def if_split_dataset(self):
        split_train = True
        batches = glob(os.path.join(self.batch_path, '*.npz'))
        if self.config.CONFIG_CV["MODE"] == 'DEBUG' and len(batches) > 0:
            split_train = False

        return split_train

    def compile_model(self, model):
        loss, raw_metrics = self.get_parameters_for_compile()
        METRICS, WEIGHTED_METRICS = self.fill_metrics(raw_metrics)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.CONFIG_TRAINER["LEARNING_RATE"]),
            loss=loss,
            metrics=METRICS,
            weighted_metrics=WEIGHTED_METRICS
        )

        return model

    def fill_metrics(self, raw_metrics):
        METRICS, WEIGHTED_METRICS = [], []

        if self.config.CONFIG_TRAINER["WITH_SAMPLE_WEIGHTS"]:
            WEIGHTED_METRICS = raw_metrics.copy()
            METRICS = None
        else:
            WEIGHTED_METRICS = None
            METRICS = raw_metrics.copy()

        return METRICS, WEIGHTED_METRICS

    def get_datasets(self, for_tuning=False):
        root_data_paths = self.dataset.get_paths(root_paths=self.config.CONFIG_PATHS[PK.SHUFFLED_PATH])
        self.batch_path = os.path.join(self.config.CONFIG_PATHS[PK.BATCHED_PATH], "")
        if len(self.except_cv_names) > 0:
            self.batch_path += self.except_cv_names[0]
        else:
            self.batch_path += "batches"
        if for_tuning:
            self.batch_path += "_" + TUNE
            ds = DistributionsChecker(paths=root_data_paths, dataset=self.dataset,
                                      config=self.config.CONFIG_DISTRIBUTION)
            tuning_index = ds.get_small_database_for_tuning()
            root_data_paths = [root_data_paths[tuning_index]]

        self.save_except_names(except_names=self.except_valid_names)

        train_ds, valid_ds = self.dataset.get_datasets(ds_paths=root_data_paths, train_names=self.except_train_names,
                                                       valid_names=self.except_valid_names,
                                                       labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                                       batch_path=self.batch_path)

        print("--- Calc class weights ---")
        class_weights = get_cw_from_meta(files=root_data_paths, labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                         names=self.except_train_names)
        print(f"---Class weights---\n{class_weights}")
        # TODO class_weights dirty fix
        class_weights = {k: v for k, v in enumerate(class_weights.values())}
        return train_ds, valid_ds, class_weights

    def get_callbacks(self):
        checkpoint_path = os.path.join(self.log_dir, self.config.CONFIG_PATHS[PK.CHECKPOINT_PATH], "cp-{epoch:04d}")

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config.CONFIG_TRAINER[TK.MODEL_CHECKPOINT]["monitor"],
            verbose=1,
            save_best_only=self.config.CONFIG_TRAINER[TK.MODEL_CHECKPOINT]["save_best_only"],
            mode=self.config.CONFIG_TRAINER[TK.MODEL_CHECKPOINT]["mode"])

        callbacks_ = [checkpoints_callback]

        if self.config.CONFIG_CV[CVK.MODE] == 'DEBUG':
            custom_callback = CustomTensorboardCallback(process=psutil.Process(os.getpid()))
            callbacks_.append(custom_callback)

        if self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["enable"]:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["monitor"],
                mode=self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["mode"],
                min_delta=self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["min_delta"],
                patience=self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["patience"],
                verbose=1,
                restore_best_weights=self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["restore_best_weights"])

            callbacks_.append(early_stopping_callback)

        return callbacks_

    def save_history(self, history):
        np.save(os.path.join(self.log_dir, HISTORY_FILE), history.history)

    def train(self):
        try:
            if self.config.CONFIG_PATHS[PK.MODE] == "WITH_GPU":
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
            elif self.config.CONFIG_PATHS[PK.MODE] != "WITHOUT_GPU":
                print(f"ERROR Mode: {self.config.CONFIG_PATHS[PK.MODE]} not available! Continue without GPU strategy")
        except Exception as e:
            self.config.telegram.send_tg_message(f'ERROR!!!, training {self.log_dir} has finished with error {e}')
            raise e  # TODO REMOVE!!

        model, history = self.train_process()

        checkpoints_paths = os.path.join(self.log_dir, self.config.CONFIG_PATHS[PK.CHECKPOINT_PATH])
        if not os.path.exists(checkpoints_paths):
            os.mkdir(checkpoints_paths)

        if not self.config.CONFIG_TRAINER[TK.EARLY_STOPPING]["enable"]:
            final_model_save_path = os.path.join(checkpoints_paths, f'cp-{len(history.history["loss"]):04d}')
            if not os.path.exists(final_model_save_path):
                os.mkdir(final_model_save_path)
            model.save(final_model_save_path)

        return model, history

    def get_output_shape(self) -> Tuple[int]:
        paths = self.dataset.get_paths(root_paths=self.config.CONFIG_PATHS[PK.SHUFFLED_PATH])
        return self.dataset.get_meta_shape(paths=paths)


if __name__ == '__main__':
    pass
