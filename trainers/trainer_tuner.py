"""import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, 'utils')
sys.path.insert(1, 'data_utils')
sys.path.insert(2, 'models')"""

import config
import keras_tuner as kt
from datetime import datetime
from tensorflow import keras

import os

import provider
from trainer_base import Trainer
from models.keras_tuner_model import KerasTunerModel
import pickle


class TrainerTuner(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuner_dir = os.path.join(self.log_dir, 'tuner')

    def restore_tuner(self, directory=''):
        if directory == '':
            directory = self.tuner_dir

        with open(os.path.join(directory, 'model.pickle'), 'rb') as handle:
            model = pickle.load(handle)

        with open(os.path.join(directory, 'params.pickle'), 'rb') as handle:
            params = pickle.load(handle)

        return self.get_tuner(model,
                              objective=params['objective'],
                              overwrite=False,
                              directory=directory)

    @staticmethod
    def get_tuner(model, *args, **kwargs):
        if config.TUNER_CLASS == 'RandomSearch':
            return kt.RandomSearch(model, *args, **kwargs)
        if config.TUNER_CLASS == 'BayesianOptimization':
            return kt.BayesianOptimization(model, *args, **kwargs)
        if config.TUNER_CLASS == 'Hyperband':
            return kt.Hyperband(model, *args, **kwargs)
        raise ValueError("config.TUNER_CLASS was wrongly written = doesn't correspond to any of 'RandomSearch',"
                         "'BayesianOptimization' or 'Hyperband'")

    @staticmethod
    def save_tuner_params(model, **kwargs):
        path = kwargs['directory']
        with open(os.path.join(path, 'model.pickle'), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'params.pickle'), 'wb') as handle:
            pickle.dump(kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compile_model(self, model):  # in this case - tune model

        params = {
            'objective': config.TUNER_OBJECTIVE,
            'max_trials': config.TUNER_MAX_TRIALS,
            'overwrite': config.TUNER_OVERWRITE,
            'directory': self.tuner_dir,
        }

        tuner = self.get_tuner(model, **params)

        tuner.search_space_summary()

        train_dataset_t, valid_dataset_t, _, class_weights_t = self.get_datasets(for_tuning=True)

        tuner.search(x=train_dataset_t,
                     epochs=config.TUNER_EPOCHS,
                     validation_data=valid_dataset_t,
                     class_weight=class_weights_t,
                     callbacks=[keras.callbacks.TensorBoard(self.tuner_dir)])

        TrainerTuner.save_tuner_params(model, **params)

        return tuner

    def get_model(self):
        model = provider.get_keras_tuner_model()
        tuner = self.compile_model(model)
        tuner.results_summary()

        hypermodel = KerasTunerModel()
        best_hp = tuner.get_best_hyperparameters()[0]
        model = hypermodel.build(best_hp)

        return best_hp, model

    def train_process(self, mirrored_strategy=None):
        self.mirrored_strategy = mirrored_strategy
        self.logging_and_copying()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, _, class_weights = self.get_datasets()

        '''-------CALLBACKS---------'''

        callbacks_ = self.get_callbacks()

        '''-------MODEL---------'''

        best_hp, model = self.get_model()

        '''-------TRAINING---------'''

        history = KerasTunerModel().fit(best_hp,
                                        model,
                                        x=train_dataset,
                                        validation_data=valid_dataset,
                                        verbose=2,
                                        epochs=config.EPOCHS,
                                        callbacks=callbacks_,
                                        use_multiprocessing=True,
                                        class_weight=class_weights,
                                        workers=int(os.cpu_count())
                                        )

        self.save_history(history)

        return model, history


if __name__ == '__main__':
    trainer = TrainerTuner()

    hypermodel_ = KerasTunerModel()

    tuner_ = trainer.restore_tuner(
        directory='C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments\\tuner_results',
        project_name='inception_3d_17.02.2022-18_20_06')

    best_hp_ = tuner_.get_best_hyperparameters()[0]
    print(best_hp_)
    model_ = hypermodel_.build(best_hp_)

    print(tuner_.results_summary(2))
    # train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
