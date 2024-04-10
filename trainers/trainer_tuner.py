import keras_tuner as kt
import tensorflow.keras as keras

import os

from trainers.trainer_base import Trainer
import pickle
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK
from configuration.parameter import (
    MODEL_BATCH_SIZE
)


class TrainerTuner(Trainer):
    def get_parameters_for_compile(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tuner_dir = os.path.join(self.log_dir, "tuner")

    def train_process(self):
        self.logging_and_copying()

        '''-------TUNER---------'''

        base_model, tuner = self.get_tuner(self.mirrored_strategy)
        best_hp, best_model = self.search(tuner)

        '''-------DATASET---------'''

        train_dataset, valid_dataset, class_weights = self.get_datasets(
            for_tuning=self.config.CONFIG_TRAINER[TK.USE_SMALLER_DATASET])

        '''-------TRAINING---------'''

        history = base_model.fit(hp=best_hp,
                                 model=best_model,
                                 x=train_dataset,
                                 validation_data=valid_dataset,
                                 verbose=2,
                                 epochs=self.config.CONFIG_TRAINER[TK.EPOCHS],
                                 batch_size=MODEL_BATCH_SIZE,
                                 callbacks=self.get_callbacks(),
                                 use_multiprocessing=True,
                                 class_weight=class_weights,
                                 workers=int(os.cpu_count())
                                 )

        self.save_history(history)
        self.dataset.delete_batches(batch_path=self.batch_path)

        return best_model, history

    def restore_tuner(self, directory=''):
        if directory == '':
            directory = self.tuner_dir

        with open(os.path.join(directory, "model.pickle"), "rb") as handle:
            model = pickle.load(handle)

        with open(os.path.join(directory, "params.pickle"), "rb") as handle:
            params = pickle.load(handle)

        params["objective"] = kt.Objective(**params["objective"])

        return model, params

    @staticmethod
    def save_tuner_params(model, **kwargs):
        path = kwargs["directory"]
        if not os.path.exists(path):
            os.mkdir(path)

        with open(os.path.join(path, "model.pickle"), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "params.pickle"), "wb") as handle:
            pickle.dump(kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_params(self):
        params_obj = self.config.CONFIG_TRAINER[TK.TUNER_PARAMS].copy()
        params_obj["directory"] = self.tuner_dir
        params_obj["objective"] = kt.Objective(**params_obj["objective"])

        return params_obj

    def search(self, tuner: kt.Tuner) -> tuple[kt.HyperParameters, keras.Model]:
        print("------ Start search tuning parameter ---------")
        print("---- Get tuning datasets ----")
        train_dataset_t, valid_dataset_t, class_weights_t = self.get_datasets(for_tuning=True)

        tuner.search(x=train_dataset_t,
                     epochs=self.config.CONFIG_TRAINER[TK.TUNER_EPOCHS],
                     validation_data=valid_dataset_t,
                     class_weight=class_weights_t,
                     callbacks=[keras.callbacks.TensorBoard(self.tuner_dir)])

        print("------ Finish search tuning parameter ---------")
        tuner.results_summary(num_trials=1)

        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]

        self.dataset.delete_batches(batch_path=self.batch_path)

        return best_hp, best_model

    def get_tuner(self, mirrored_strategy=None) -> tuple[kt.HyperModel, kt.Tuner]:
        if self.config.CONFIG_TRAINER[TK.RESTORE]:
            base_model, params = self.restore_tuner()
        else:
            base_model = self.get_model()
            params = self.get_params()
            TrainerTuner.save_tuner_params(base_model, **params)

        return base_model, self.config.CONFIG_TRAINER[TK.TUNER](base_model, **params,
                                                                distribution_strategy=mirrored_strategy)

    def get_model(self) -> kt.HyperModel:
        base_model = self.config.CONFIG_TRAINER[TK.MODEL](input_shape=self.get_output_shape(),
                                                          model_config=self.config.CONFIG_TRAINER[TK.MODEL_CONFIG],
                                                          num_of_labels=len(self.config.CONFIG_DATALOADER[
                                                                                DLK.LABELS_TO_TRAIN]),
                                                          custom_metrics=self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS])

        return base_model


if __name__ == '__main__':
    '''tuner_ = trainer.restore_tuner(
        directory='C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments\\tuner_results',
        project_name='inception_3d_17.02.2022-18_20_06')

    best_hp_ = tuner_.get_best_hyperparameters()[0]
    print(best_hp_)
    model_ = hypermodel_.build(best_hp_)

    print(tuner_.results_summary(2))'''
    # train(except_indexes=['2019_09_04_12_43_40_',
    # '2020_05_28_15_20_27_', '2019_07_12_11_15_49_',
    # '2020_05_15_12_43_58_'])
