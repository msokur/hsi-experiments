import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import abc
import pickle

from data_utils.generator import DataGenerator
from data_utils.data_archive.data_archive import DataArchive
from configuration.copy_py_files import copy_files
from configuration.get_config import telegram
from configuration.keys import TrainerKeys as TK, PathKeys as PK, DataLoaderKeys as DLK
from configuration.parameter import (
    VALID_LOG, HISTORY_FILE, TUNE, TRAIN, VALID
)


class Trainer:
    def __init__(self, data_archive: DataArchive, config_trainer: dict, config_paths: dict, config_dataloader: dict,
                 model_name: str, except_indexes=None, valid_except_indexes=None):
        self.data_archive = data_archive
        self.CONFIG_TRAINER = config_trainer
        self.CONFIG_PATHS = config_paths
        self.CONFIG_DATALOADER = config_dataloader
        if valid_except_indexes is None:
            valid_except_indexes = []
        if except_indexes is None:
            except_indexes = []
        self.batch_path = None
        self.mirrored_strategy = None
        self.log_dir = model_name
        self.excepted_indexes = except_indexes.copy()
        self.valid_except_indexes = valid_except_indexes.copy()

    @abc.abstractmethod
    def train_process(self):
        pass

    def save_valid_except_indexes(self, valid_except_indexes):
        with open(os.path.join(self.log_dir, VALID_LOG), "wb") as f:
            pickle.dump(valid_except_indexes, f, pickle.HIGHEST_PROTOCOL)

    def logging_and_copying(self):
        if not self.CONFIG_TRAINER[TK.RESTORE]:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            copy_files(self.log_dir, self.CONFIG_TRAINER["FILES_TO_COPY"])

    def get_datasets(self, for_tuning=False):
        # train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)
        self.batch_path = self.CONFIG_PATHS[PK.BATCHED_PATH]
        if len(self.excepted_indexes) > 0:
            self.batch_path += '_' + self.excepted_indexes[0]
            if for_tuning:
                self.batch_path += TUNE
        if not os.path.exists(self.batch_path):
            os.mkdir(self.batch_path)

        train_generator = DataGenerator(TRAIN,
                                        self.CONFIG_PATHS[PK.SHUFFLED_PATH],
                                        self.batch_path,
                                        batch_size=self.CONFIG_TRAINER[TK.BATCH_SIZE],
                                        split_factor=self.CONFIG_TRAINER[TK.SPLIT_FACTOR],
                                        split_flag=True,
                                        valid_except_indexes=self.valid_except_indexes.copy(),
                                        except_indexes=self.excepted_indexes.copy(),
                                        for_tuning=for_tuning,
                                        log_dir=self.log_dir)
        self.save_valid_except_indexes(train_generator.valid_except_indexes)
        valid_generator = DataGenerator(VALID,
                                        self.CONFIG_PATHS[PK.SHUFFLED_PATH],
                                        self.batch_path,
                                        batch_size=self.CONFIG_TRAINER[TK.BATCH_SIZE],
                                        split_factor=self.CONFIG_TRAINER[TK.SPLIT_FACTOR],
                                        split_flag=False,
                                        except_indexes=self.excepted_indexes,
                                        valid_except_indexes=train_generator.valid_except_indexes,
                                        for_tuning=for_tuning,
                                        log_dir=self.log_dir)

        class_weights = train_generator.get_class_weights()
        print(class_weights)

        def gen_train_generator():
            for i in range(train_generator.len):
                yield train_generator.getitem(i)

        train_dataset = tf.data.Dataset.from_generator(gen_train_generator,
                                                       output_signature=self.__get_output_signature())

        def gen_valid_generator():
            for i in range(valid_generator.len):
                yield valid_generator.getitem(i)

        valid_dataset = tf.data.Dataset.from_generator(gen_valid_generator,
                                                       output_signature=self.__get_output_signature())

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        valid_dataset = valid_dataset.with_options(options)

        return train_dataset, valid_dataset, train_generator, class_weights

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

        # send_tg_message_history(self.log_dir, history)

        return model, history

    def get_output_shape(self):
        if DLK.OUTPUT_SIGNATURE in self.CONFIG_DATALOADER:
            shape_spec = self.CONFIG_DATALOADER[DLK.OUTPUT_SIGNATURE]
        else:
            shape_spec = self.CONFIG_DATALOADER[DLK.LAST_NM] - self.CONFIG_DATALOADER[DLK.FIRST_NM]

        if self.CONFIG_DATALOADER[DLK.D3]:
            output_shape = (self.CONFIG_DATALOADER[DLK.D3_SIZE][0], self.CONFIG_DATALOADER[DLK.D3_SIZE][1], shape_spec)
        else:
            output_shape = (shape_spec,)

        return output_shape

    def __get_output_signature(self):
        shape = self.get_output_shape()

        output_signature = (
            tf.TensorSpec(shape=((None,) + shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32))

        if self.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS]:
            output_signature += (tf.TensorSpec(shape=(None,), dtype=tf.float32),)

        return output_signature


if __name__ == '__main__':
    pass
