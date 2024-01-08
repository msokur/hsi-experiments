import tensorflow as tf
from tensorflow import keras
import os
from shutil import rmtree
import numpy as np
import abc
import pickle
import psutil
from glob import glob

import data_utils.generator as generator
from configuration.copy_py_files import copy_files
from configuration import get_config as conf
from callbacks import CustomTensorboardCallback


class Trainer:
    def __init__(self, model_name: str, except_indexes=None, valid_except_indexes=None):
        self.CONFIG_TRAINER = conf.CONFIG_TRAINER
        self.CONFIG_PATHS = conf.CONFIG_PATHS
        self.CONFIG_DATALOADER = conf.CONFIG_DATALOADER
        self.CONFIG_CV = conf.CONFIG_CV
        
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
    def get_model(self):
        pass
    
    @abc.abstractmethod
    def get_parameters_for_compile(self):
        #shold return Loss and Metrics
        pass
    
    def compile_model(self, model):
        loss, raw_metrics = self.get_parameters_for_compile()
        METRICS, WEIGHTED_METRICS = self.fill_metrics(raw_metrics)
        print(loss, METRICS, WEIGHTED_METRICS)
            
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.CONFIG_TRAINER["LEARNING_RATE"]),
            loss=loss,
            metrics=METRICS,
            weighted_metrics=WEIGHTED_METRICS
        )

        return model
    
    def fill_metrics(self, raw_metrics):
        METRICS, WEIGHTED_METRICS = [], []
        
        if self.CONFIG_TRAINER["WITH_SAMPLE_WEIGHTS"]:
            WEIGHTED_METRICS = raw_metrics.copy()
            METRICS = None
        else:
            WEIGHTED_METRICS = None
            METRICS = raw_metrics.copy()
            
        return METRICS, WEIGHTED_METRICS
        
    def save_valid_except_indexes(self, valid_except_indexes):
        with open(os.path.join(self.log_dir, "valid.valid_except_names"), "wb") as f:
            pickle.dump(valid_except_indexes, f, pickle.HIGHEST_PROTOCOL)

    def logging_and_copying(self):
        if self.CONFIG_TRAINER["TYPE"] != "Restore":
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            copy_files(self.log_dir, self.CONFIG_TRAINER["FILES_TO_COPY"], self.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])
    
    def if_split_dataset(self)
        split_train=True
        batches = glob(os.path.join(self.batch_path, '*.npz'))
        if self.CONFIG_CV["MODE"] == 'DEBUG' and len(batches) > 0:
            split_train=False
            
        return split_train
    
    def get_datasets(self, for_tuning=False):
        # train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)
        self.batch_path = self.CONFIG_PATHS["BATCHED_PATH"]
        if len(self.excepted_indexes) > 0:
            self.batch_path += '_' + self.excepted_indexes[0]
            if for_tuning:
                self.batch_path += "tune"
        if not os.path.exists(self.batch_path):
            os.mkdir(self.batch_path)
            
        split_train=self.if_split_dataset()
            
        train_generator = generator.DataGenerator("train",
                                                  self.CONFIG_PATHS["SHUFFLED_PATH"],
                                                  self.batch_path,
                                                  batch_size=self.CONFIG_TRAINER["BATCH_SIZE"],
                                                  split_factor=self.CONFIG_TRAINER["SPLIT_FACTOR"],
                                                  split_flag=split_train,
                                                  valid_except_indexes=self.valid_except_indexes.copy(),
                                                  except_indexes=self.excepted_indexes.copy(),
                                                  for_tuning=for_tuning,
                                                  log_dir=self.log_dir)
        self.save_valid_except_indexes(train_generator.valid_except_indexes)
        valid_generator = generator.DataGenerator("valid",
                                                  self.CONFIG_PATHS["SHUFFLED_PATH"],
                                                  self.batch_path,
                                                  batch_size=self.CONFIG_TRAINER["BATCH_SIZE"],
                                                  split_factor=self.CONFIG_TRAINER["SPLIT_FACTOR"],
                                                  split_flag=False,
                                                  except_indexes=self.excepted_indexes,
                                                  valid_except_indexes=train_generator.valid_except_indexes,
                                                  for_tuning=for_tuning,
                                                  log_dir=self.log_dir)

        class_weights = train_generator.get_class_weights()
        print('Class weights:', class_weights)

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
        checkpoint_path = os.path.join(self.log_dir, self.CONFIG_PATHS["CHECKPOINT_PATH"], "cp-{epoch:04d}")
        

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.CONFIG_TRAINER["MODEL_CHECKPOINT"]["monitor"],
            verbose=1,
            save_best_only=self.CONFIG_TRAINER["MODEL_CHECKPOINT"]["save_best_only"],
            mode=self.CONFIG_TRAINER["MODEL_CHECKPOINT"]["mode"])
        
        callbacks_ = [checkpoints_callback]
        
        if self.CONFIG_CV["MODE"] == 'DEBUG':
            custom_callback = CustomTensorboardCallback(process=psutil.Process(os.getpid()))
            callbacks_.append(custom_callback)
            
        if self.CONFIG_TRAINER["EARLY_STOPPING"]["enable"]:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=self.CONFIG_TRAINER["EARLY_STOPPING"]["monitor"],
                mode=self.CONFIG_TRAINER["EARLY_STOPPING"]["mode"],
                min_delta=self.CONFIG_TRAINER["EARLY_STOPPING"]["min_delta"],
                patience=self.CONFIG_TRAINER["EARLY_STOPPING"]["patience"],
                verbose=1,
                restore_best_weights=self.CONFIG_TRAINER["EARLY_STOPPING"]["restore_best_weights"])

            callbacks_.append(early_stopping_callback)

        return callbacks_

    def save_history(self, history):
        np.save(os.path.join(self.log_dir, 'history'), history.history)

    def train_process(self, mirrored_strategy=None):
        self.mirrored_strategy = mirrored_strategy
        self.logging_and_copying()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, train_generator, class_weights = self.get_datasets(
            for_tuning=self.CONFIG_TRAINER["SMALLER_DATASET"])

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
            epochs=self.CONFIG_TRAINER["EPOCHS"],
            verbose=2,
            initial_epoch=initial_epoch,
            batch_size=self.CONFIG_TRAINER["BATCH_SIZE"],
            callbacks=callbacks_,
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))

        self.save_history(history)
        
        if self.CONFIG_CV["MODE"] == 'RUN':
            rmtree(self.batch_path)

        return model, history

    def train(self):
        try:
            if self.CONFIG_PATHS["MODE"] == 'LOCAL_GPU' or self.CONFIG_PATHS["MODE"] == 'CLUSTER':
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
                # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
                try:
                    with mirrored_strategy.scope():
                        model, history = self.train_process(mirrored_strategy=mirrored_strategy)
                except IndexError:
                    pass
            else:
                model, history = self.train_process()

        except Exception as e:
            conf.telegram.send_tg_message(f'ERROR!!!, training {self.log_dir} has finished with error {e}')
            raise e  # TODO REMOVE!!

        checkpoints_paths = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.exists(checkpoints_paths):
            os.mkdir(checkpoints_paths)

        if not self.CONFIG_TRAINER["EARLY_STOPPING"]["enable"]:
            final_model_save_path = os.path.join(self.log_dir, 'checkpoints', f'cp-{len(history.history["loss"]):04d}')
            if not os.path.exists(final_model_save_path):
                os.mkdir(final_model_save_path)
            model.save(final_model_save_path)

        # send_tg_message_history(self.log_dir, history)

        return model

    def get_output_shape(self):
        if "OUTPUT_SIGNATURE_X_FEATURES" in self.CONFIG_DATALOADER:
            shape_spec = self.CONFIG_DATALOADER["OUTPUT_SIGNATURE_X_FEATURES"]
        else:
            shape_spec = self.CONFIG_DATALOADER["LAST_NM"] - self.CONFIG_DATALOADER["FIRST_NM"]

        if self.CONFIG_DATALOADER["3D"]:
            output_shape = (self.CONFIG_DATALOADER["3D_SIZE"][0], self.CONFIG_DATALOADER["3D_SIZE"][1], shape_spec)
        else:
            output_shape = (shape_spec,)

        return output_shape

    def __get_output_signature(self):
        shape = self.get_output_shape()

        output_signature = (
            tf.TensorSpec(shape=((None,) + shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32))

        if self.CONFIG_TRAINER["WITH_SAMPLE_WEIGHTS"]:
            output_signature += (tf.TensorSpec(shape=(None,), dtype=tf.float32),)

        return output_signature


if __name__ == '__main__':
    pass
