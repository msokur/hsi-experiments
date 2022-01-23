import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, 'utils')
sys.path.insert(1, 'data_utils')
sys.path.insert(2, 'models')

import tensorflow as tf
from tensorflow import keras
import config
import keras_tuner as kt
from datetime import datetime

import model
import model_3d
import callbacks
import os
from shutil import copyfile
import telegram_send
from generator import DataGenerator
import tf_metrics
import numpy as np
import psutil
from trainer_base import Trainer
from models.keras_tuner_model import KerasTunerModel

from utils import send_tg_message, send_tg_message_history


class TrainerTuner(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model):  # in this case - tune model

        def get_tuner(*args, **kwargs):
            if config.TUNER_CLASS == 'RandomSearch':
                return kt.RandomSearch(*args, **kwargs)
            if config.TUNER_CLASS == 'BayesianOptimization':
                return kt.BayesianOptimization(*args, **kwargs)
            if config.TUNER_CLASS == 'Hyperband':
                return kt.Hyperband(*args, **kwargs)
            raise ValueError("config.TUNER_CLASS was wrongly written = doesn't correspond to any of 'RandomSearch',"
                             "'BayesianOptimization' or 'Hyperband'")

        project_name = config.TUNER_PROJECT_NAME
        if config.TUNER_ADD_TIME:
            project_name += datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

        tuner = get_tuner(
            KerasTunerModel(),
            objective=config.TUNER_OBJECTIVE,
            max_trials=config.TUNER_MAX_TRIALS,
            overwrite=config.TUNER_OVERWRITE,
            directory=config.TUNER_DIRECTORY,
            project_name=project_name,
        )
        tuner.search_space_summary()

        train_dataset_t, valid_dataset_t, _, class_weights_t = self.get_datasets(for_tuning=True)

        tuner.search(x=train_dataset_t,
                     epochs=config.TUNER_EPOCHS,
                     validation_data=valid_dataset_t,
                     class_weight=class_weights_t)
        return tuner

    def get_model(self):
        tuner = self.compile_model(None)
        tuner.results_summary()

        hypermodel = KerasTunerModel()
        best_hp = tuner.get_best_hyperparameters()[0]
        model = hypermodel.build(best_hp)

        return best_hp, model

    def __train(self, mirrored_strategy=None):
        self.mirrored_strategy = mirrored_strategy
        self.logging_and_copying()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, _, class_weights = self.get_datasets()

        '''-------CALLBACKS---------'''

        callbacks_ = self.get_callbacks()

        '''-------MODEL---------'''

        best_hp, model = self.get_model()

        '''-------TRAINING---------'''

        model, history = KerasTunerModel(best_hp,
                                         model,
                                         x=train_dataset,
                                         validation_data=valid_dataset,
                                         verbose=2,
                                         callbacks=callbacks_,
                                         use_multiprocessing=True,
                                         class_weight=class_weights,
                                         workers=int(os.cpu_count())
                                         )

        '''history = model.fit(
            # x=train_generator,
            # validation_data=valid_generator,
            x=train_dataset,
            validation_data=valid_dataset,
            ?  epochs=config.EPOCHS,
            verbose=2,
            ?  batch_size=config.BATCH_SIZE,
            callbacks=callbacks_,
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))'''

        np.save(os.path.join(self.log_dir, 'history.history'), history.history)

        return model, history


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    # train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
