import tensorflow as tf
from tensorflow import keras
import os
from shutil import rmtree
import numpy as np
import abc
import pickle

import data_utils.generator as generator
from configuration.copy_py_files import copy_files
from configuration import get_config as conf


class Trainer:
    def __init__(self, model_name: str, except_indexes=None, valid_except_indexes=None):
        self.trainer = conf.TRAINER
        self.paths = conf.PATHS
        self.loader = conf.DATALOADER
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
    def compile_model(self, model):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    def save_valid_except_indexes(self, valid_except_indexes):
        with open(os.path.join(self.log_dir, "valid.valid_except_names"), "wb") as f:
            pickle.dump(valid_except_indexes, f, pickle.HIGHEST_PROTOCOL)

    def logging_and_copying(self):
        if self.trainer["TYPE"] != "Restore":
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            copy_files(self.log_dir, self.trainer["FILES_TO_COPY"], self.paths["SYSTEM_PATHS_DELIMITER"])

    def get_datasets(self, for_tuning=False):
        # train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)
        self.batch_path = self.paths["BATCHED_PATH"]
        if len(self.excepted_indexes) > 0:
            self.batch_path += '_' + self.excepted_indexes[0]
        if not os.path.exists(self.batch_path):
            os.mkdir(self.batch_path)

        train_generator = generator.DataGenerator("train",
                                                  self.paths["SHUFFLED_PATH"],
                                                  self.batch_path,
                                                  batch_size=self.trainer["BATCH_SIZE"],
                                                  split_factor=self.trainer["SPLIT_FACTOR"],
                                                  split_flag=True,
                                                  valid_except_indexes=self.valid_except_indexes.copy(),
                                                  except_indexes=self.excepted_indexes.copy(),
                                                  for_tuning=for_tuning,
                                                  log_dir=self.log_dir)
        self.save_valid_except_indexes(train_generator.valid_except_indexes)
        valid_generator = generator.DataGenerator("valid",
                                                  self.paths["SHUFFLED_PATH"],
                                                  self.batch_path,
                                                  batch_size=self.trainer["BATCH_SIZE"],
                                                  split_factor=self.trainer["SPLIT_FACTOR"],
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
        checkpoint_path = os.path.join(self.log_dir, self.paths["CHECKPOINT_PATH"], "cp-{epoch:04d}")

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.trainer["MODEL_CHECKPOINT"]["monitor"],
            verbose=1,
            save_best_only=self.trainer["MODEL_CHECKPOINT"]["save_best_only"],
            mode=self.trainer["MODEL_CHECKPOINT"]["mode"])

        callbacks_ = [checkpoints_callback]
        if self.trainer["EARLY_STOPPING"]["enable"]:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=self.trainer["EARLY_STOPPING"]["monitor"],
                mode=self.trainer["EARLY_STOPPING"]["mode"],
                min_delta=self.trainer["EARLY_STOPPING"]["min_delta"],
                patience=self.trainer["EARLY_STOPPING"]["patience"],
                verbose=1,
                restore_best_weights=self.trainer["EARLY_STOPPING"]["restore_best_weights"])

            callbacks_.append(early_stopping_callback)

        return callbacks_

    def save_history(self, history):
        np.save(os.path.join(self.log_dir, 'history'), history.history)

    def train_process(self, mirrored_strategy=None):
        self.mirrored_strategy = mirrored_strategy
        self.logging_and_copying()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, train_generator, class_weights = self.get_datasets(
            for_tuning=self.trainer["SMALLER_DATASET"])

        '''-------CALLBACKS---------'''

        callbacks_ = self.get_callbacks()

        '''-------MODEL---------'''

        model, initial_epoch = self.get_model()
        model.summary()

        '''-------TRAINING---------'''

        history = model.fit(
            # x=train_generator,
            # validation_data=valid_generator,
            x=train_dataset,
            validation_data=valid_dataset,
            epochs=self.trainer["EPOCHS"],
            verbose=2,
            initial_epoch=initial_epoch,
            batch_size=self.trainer["BATCH_SIZE"],
            callbacks=callbacks_,
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))

        self.save_history(history)

        rmtree(self.batch_path)

        return model, history

    def train(self):
        try:
            if self.paths["MODE"] == 'LOCAL_GPU' or self.paths["MODE"] == 'CLUSTER':
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                    except RuntimeError as e:
                        print(e)

                mirrored_strategy = tf.distribute.experimental.CentralStorageStrategy()

                with mirrored_strategy.scope():
                    model, history = self.train_process(mirrored_strategy=mirrored_strategy)
            else:
                model, history = self.train_process()

        except Exception as e:
            conf.telegram.send_tg_message(f'ERROR!!!, training {self.log_dir} has finished with error {e}')
            raise e  # TODO REMOVE!!

        checkpoints_paths = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.exists(checkpoints_paths):
            os.mkdir(checkpoints_paths)

        if not self.trainer["EARLY_STOPPING"]["enable"]:
            final_model_save_path = os.path.join(self.log_dir, 'checkpoints', f'cp-{len(history.history["loss"]):04d}')
            if not os.path.exists(final_model_save_path):
                os.mkdir(final_model_save_path)
            model.save(final_model_save_path)

        # send_tg_message_history(self.log_dir, history)

        return model

    def __get_output_shape(self):
        if "OUTPUT_SIGNATURE_X_FEATURES" in self.loader:
            shape_spec = self.loader["OUTPUT_SIGNATURE_X_FEATURES"]
        else:
            shape_spec = self.loader["LAST_NM"] - self.loader["FIRST_NM"]

        if self.loader["3D"]:
            output_shape = (self.loader["3D_SIZE"][0], self.loader["3D_SIZE"][1], shape_spec)
        else:
            output_shape = (shape_spec,)

        return output_shape

    def __get_output_signature(self):
        shape = self.__get_output_shape()

        output_signature = (
            tf.TensorSpec(shape=((None,) + shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32))

        if self.trainer["WITH_SAMPLE_WEIGHTS"]:
            output_signature += (tf.TensorSpec(shape=(None,), dtype=tf.float32),)

        return output_signature


if __name__ == '__main__':
    pass
