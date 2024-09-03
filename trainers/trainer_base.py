from typing import Tuple, List, Dict

import tensorflow as tf
import os
import numpy as np
import abc
import pickle

from data_utils.dataset.meta_files import get_class_weights_from_meta
from provider import get_dataset
from .utils import get_callbacks, CVStepNames

from data_utils.data_storage import DataStorage
from configuration.copy_py_files import copy_files
from configuration.keys import (
    TrainerKeys as TK,
    PathKeys as PK,
    DataLoaderKeys as DLK,
    CrossValidationKeys as CVK
)
from configuration.parameter import (
    DATASET_TYPE, FILE_WITH_VALID_NAME, HISTORY_FILE
)


class Trainer:
    def __init__(self, config, data_storage: DataStorage, log_dir: str):
        self.config = config
        self.data_storage = data_storage
        self.dataset = get_dataset(typ=DATASET_TYPE, config=config, data_storage=self.data_storage)
        self.log_dir = log_dir
        self.mirrored_strategy = self.get_mirrored_strategy()
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def get_mirrored_strategy(self):
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

                return tf.distribute.experimental.CentralStorageStrategy()
                # self.mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
            elif self.config.CONFIG_PATHS[PK.MODE] != "WITHOUT_GPU":
                print(f"ERROR Mode: {self.config.CONFIG_PATHS[PK.MODE]} not available! Continue without GPU strategy")
                return None
        except Exception as e:
            self.config.telegram.send_tg_message(f'ERROR!!!, training {self.log_dir} has finished with error {e}')
            raise e  # TODO REMOVE!!

    def train(self, dataset_paths: list[str], train_step_names: CVStepNames, step_name: str, batch_path: str):
        train_step_dir = os.path.join(self.log_dir, step_name)

        self.logging_and_copying(store_dir=train_step_dir)

        self.save_except_names(store_dir=train_step_dir,
                               except_names=train_step_names.VALID_NAMES)

        datasets_and_class_weights = self.get_datasets(dataset_paths=dataset_paths,
                                                       train_step_names=train_step_names,
                                                       batch_path=batch_path)
        try:
            model, history = self.train_process(train_log_dir=train_step_dir,
                                                datasets=datasets_and_class_weights[0:2],
                                                class_weights=datasets_and_class_weights[2],
                                                batch_path=batch_path)
            self.save_history(train_log_dir=train_step_dir,
                              history=history)
        except Exception as exception:
            if self.config.CONFIG_CV[CVK.MODE] == "RUN":
                self.dataset.delete_batches(batch_path=batch_path)
            raise exception

        if self.config.CONFIG_CV[CVK.MODE] == "RUN":
            self.dataset.delete_batches(batch_path=batch_path)

        if not self.config.CONFIG_TRAINER[TK.CALLBACKS][TK.EARLY_STOPPING]["enable"]:
            checkpoints_paths = os.path.join(train_step_dir, self.config.CONFIG_PATHS[PK.CHECKPOINT_FOLDER])
            if not os.path.exists(checkpoints_paths):
                os.mkdir(checkpoints_paths)

            final_model_save_path = os.path.join(checkpoints_paths, f'cp-{len(history.history["loss"]):04d}')
            if not os.path.exists(final_model_save_path):
                os.mkdir(final_model_save_path)
            model.save(final_model_save_path)

        return model, history

    def logging_and_copying(self, store_dir: str):
        if not self.config.CONFIG_TRAINER[TK.RESTORE]:
            if not os.path.exists(store_dir):
                os.mkdir(store_dir)

            copy_files(store_dir, self.config.CONFIG_TRAINER["FILES_TO_COPY"])

    @abc.abstractmethod
    def train_process(self, train_log_dir: str, datasets: tuple, class_weights: Dict[int, float], batch_path: str):
        pass

    def get_datasets(self, dataset_paths: list[str], train_step_names: CVStepNames, batch_path: str):
        train_ds, valid_ds = self.dataset.get_datasets(dataset_paths=dataset_paths,
                                                       train_names=train_step_names.TRAIN_NAMES,
                                                       valid_names=train_step_names.VALID_NAMES,
                                                       labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                                       batch_path=batch_path)

        class_weights = self.get_class_weights(train_names=train_step_names.TRAIN_NAMES,
                                               dataset_paths=dataset_paths)
        return train_ds, valid_ds, class_weights

    @staticmethod
    def save_except_names(store_dir: str, except_names: List[str]):
        with open(os.path.join(store_dir, FILE_WITH_VALID_NAME), "wb") as f:
            pickle.dump(except_names, f, pickle.HIGHEST_PROTOCOL)

    def get_class_weights(self, train_names: list[str], dataset_paths: list[str]):
        class_weights = None
        if not self.config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS]:
            class_weights = get_class_weights_from_meta(files=dataset_paths,
                                                        labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                                        names=train_names)

            class_weights = {k: v for k, v in enumerate(class_weights.values())}
        print(f"---Class weights---\n{class_weights}")
        return class_weights

    def get_callbacks(self, train_log_dir: str):
        checkpoint_dir = str(os.path.join(train_log_dir, self.config.CONFIG_PATHS[PK.CHECKPOINT_FOLDER]))
        return get_callbacks(callback_configs=self.config.CONFIG_TRAINER[TK.CALLBACKS],
                             checkpoint_dir=checkpoint_dir,
                             debug=self.config.CONFIG_CV[CVK.MODE] == "DEBUG")

    @staticmethod
    def save_history(train_log_dir: str, history):
        np.save(str(os.path.join(train_log_dir, HISTORY_FILE)), history.history)

    def get_dataset_paths(self):
        return self.dataset.get_dataset_paths(root_paths=self.config.CONFIG_PATHS[PK.SHUFFLED_PATH])

    def get_output_shape(self) -> Tuple[int]:
        paths = self.get_dataset_paths()
        return self.dataset.get_meta_shape(paths=paths)
