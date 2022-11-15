import tensorflow as tf
from tensorflow import keras
import os
from shutil import copyfile, rmtree
import numpy as np
import psutil
import abc
import pickle
from glob import glob

import config
from utils import send_tg_message, send_tg_message_history
import data_utils.generator as generator
import callbacks


class Trainer:
    def __init__(self, model_name=config.MODEL_NAME, except_indexes=[], valid_except_indexes=[]):
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
        with open(os.path.join(self.log_dir, 'valid.valid_except_names'), 'wb') as f:
            pickle.dump(valid_except_indexes, f, pickle.HIGHEST_PROTOCOL)

    def logging_and_copying(self):
        if not config.RESTORE_MODEL:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            for file in config.FILES_TO_COPY:
                if os.path.exists(file):
                    copyfile(file, os.path.join(self.log_dir, file.split(config.SYSTEM_PATHS_DELIMITER)[-1]))

    def get_datasets(self, for_tuning=False):
        # train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)
        self.batch_path = config.BATCHED_PATH
        if len(self.excepted_indexes) > 0:
            self.batch_path += '_' + self.excepted_indexes[0]
        if not os.path.exists(self.batch_path):
            os.mkdir(self.batch_path)

        train_generator = generator.DataGenerator('train',
                                                  config.SHUFFLED_PATH,
                                                  # config.BATCHED_PATH,
                                                  self.batch_path,
                                                  split_flag=True,
                                                  valid_except_indexes=self.valid_except_indexes.copy(),
                                                  except_indexes=self.excepted_indexes.copy(),
                                                  for_tuning=for_tuning,
                                                  log_dir=self.log_dir)
        self.save_valid_except_indexes(train_generator.valid_except_indexes)
        valid_generator = generator.DataGenerator('valid',
                                                  config.SHUFFLED_PATH,
                                                  # config.BATCHED_PATH,
                                                  self.batch_path,
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

        train_dataset = tf.data.Dataset.from_generator(gen_train_generator, output_signature=config.OUTPUT_SIGNATURE)

        def gen_valid_generator():
            for i in range(valid_generator.len):
                yield valid_generator.getitem(i)

        valid_dataset = tf.data.Dataset.from_generator(gen_valid_generator, output_signature=config.OUTPUT_SIGNATURE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        valid_dataset = valid_dataset.with_options(options)

        return train_dataset, valid_dataset, train_generator, class_weights

    def get_callbacks(self):
        process = psutil.Process(os.getpid())

        tensorboard_callback = callbacks.CustomTensorboardCallback(
            log_dir=self.log_dir,
            # write_graph=True,
            # histogram_freq=1,
            # profile_batch = '20,30',
            except_indexes=self.excepted_indexes,
            # train_generator=train_generator, #used for gradients counting TODO fix or remove
            strategy=self.mirrored_strategy,
            process=process)

        # gradient_callback = callbacks.GradientCallback()

        checkpoint_path = os.path.join(self.log_dir, config.CHECKPOINT_PATH, 'cp-{epoch:04d}')

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.HISTORY_ARGMIN,
            verbose=1,
            save_best_only=True,
            mode='max')
        '''checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=2,
            save_freq=config.NUM_OF_PAT*config.BATCH_SIZE*config.WRITE_CHECKPOINT_EVERY_Xth_STEP)'''

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            min_delta=0,
            patience=5,
            verbose=1,
            restore_best_weights=True)

        callbacks_ = [checkpoints_callback]
        if config.WITH_EARLY_STOPPING:
            callbacks_.append(early_stopping_callback)

        return callbacks_

    def save_history(self, history):
        np.save(os.path.join(self.log_dir, 'history'), history.history)

    def train_process(self, mirrored_strategy=None):
        self.mirrored_strategy = mirrored_strategy
        self.logging_and_copying()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, train_generator, class_weights = self.get_datasets(
            for_tuning=config.WITH_SMALLER_DATASET)

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
            epochs=config.EPOCHS,
            verbose=2,
            initial_epoch=initial_epoch,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks_,
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))

        self.save_history(history)

        rmtree(self.batch_path)

        return model, history

    def train(self):
        try:
            if config.MODE == 'LOCAL_GPU' or config.MODE == 'CLUSTER':
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                    except RuntimeError as e:
                        print(e)

                # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
                # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
                mirrored_strategy = tf.distribute.experimental.CentralStorageStrategy()
                # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
                # mirrored_strategy = tf.distribute.MirroredStrategy()

                with mirrored_strategy.scope():
                    model, history = self.train_process(mirrored_strategy=mirrored_strategy)
            else:
                model, history = self.train_process()

        except Exception as e:
            send_tg_message(f'{config.USER}, ERROR!!!, training {self.log_dir} has finished with error {e}')
            raise e  # TODO REMOVE!!

        checkpoints_paths = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.exists(checkpoints_paths):
            os.mkdir(checkpoints_paths)

        final_model_save_path = os.path.join(self.log_dir, 'checkpoints', f'cp-{len(history.history["loss"]):04d}')
        if not os.path.exists(final_model_save_path):
            os.mkdir(final_model_save_path)
        model.save(final_model_save_path)

        # send_tg_message_history(self.log_dir, history)

        return model


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    # train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
