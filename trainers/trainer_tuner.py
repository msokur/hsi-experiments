from shutil import rmtree

import keras_tuner as kt
from tensorflow import keras

import os

from trainers.trainer_base import Trainer
import pickle


class TrainerTuner(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tuner_dir = os.path.join(self.log_dir, "tuner")

    def restore_tuner(self, directory=''):
        if directory == '':
            directory = self.tuner_dir

        with open(os.path.join(directory, "model.pickle"), "rb") as handle:
            model = pickle.load(handle)

        with open(os.path.join(directory, "params.pickle"), "rb") as handle:
            params = pickle.load(handle)

        return self.trainer["TUNER"](model,
                                     objective=kt.Objective(**params["objective"]),
                                     overwrite=False,
                                     directory=directory)

    @staticmethod
    def save_tuner_params(model, **kwargs):
        path = kwargs["directory"]
        with open(os.path.join(path, "model.pickle"), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "params.pickle"), "wb") as handle:
            pickle.dump(kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compile_model(self, model):  # in this case - tune model

        params = {
            "objective": self.trainer["TUNER_OBJECTIVE"],
            "max_trials": self.trainer["TUNER_MAX_TRIALS"],
            "overwrite": self.trainer["TUNER_OVERWRITE"],
            "directory": self.tuner_dir,
        }
        params_obj = params.copy()
        params_obj["objective"] = kt.Objective(**self.trainer["TUNER_OBJECTIVE"])

        tuner = self.trainer["TUNER"](model, **params_obj)

        tuner.search_space_summary()

        train_dataset_t, valid_dataset_t, _, class_weights_t = self.get_datasets(for_tuning=True)

        tuner.search(x=train_dataset_t,
                     epochs=self.trainer["TUNER_EPOCHS"],
                     validation_data=valid_dataset_t,
                     class_weight=class_weights_t,
                     callbacks=[keras.callbacks.TensorBoard(self.tuner_dir)])

        TrainerTuner.save_tuner_params(model, **params)

        rmtree(self.batch_path)

        return tuner

    def get_model(self):
        print("------ Start search tuning parameter ---------")
        base_model = self.trainer["MODEL"](shape=self.get_output_shape(), conf=self.trainer["MODEL_CONFIG"],
                                           num_of_labels=len(self.loader["LABELS_TO_TRAIN"]))
        tuner = self.compile_model(base_model)
        print("------ End search tuning parameter ---------")

        tuner.results_summary()

        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]

        return best_hp, best_model, base_model

    def train_process(self, mirrored_strategy=None):
        self.mirrored_strategy = mirrored_strategy
        self.logging_and_copying()

        '''-------CALLBACKS---------'''

        callbacks_ = self.get_callbacks()

        '''-------MODEL---------'''

        best_hp, best_model, base_model = self.get_model()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, _, class_weights = self.get_datasets()

        '''-------TRAINING---------'''

        history = base_model.fit(best_hp,
                                 best_model,
                                 x=train_dataset,
                                 validation_data=valid_dataset,
                                 verbose=2,
                                 epochs=self.trainer["EPOCHS"],
                                 callbacks=callbacks_,
                                 use_multiprocessing=True,
                                 class_weight=class_weights,
                                 workers=int(os.cpu_count())
                                 )

        self.save_history(history)

        rmtree(self.batch_path)

        return best_model, history


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
