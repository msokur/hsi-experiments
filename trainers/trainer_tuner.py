from typing import List, Dict

import keras_tuner as kt
import tensorflow.keras as keras

import os
import pickle

from trainers.trainer_interface import TrainerInterface
from .utils import get_early_stopping

from configuration.keys import (
    TrainerKeys as TK,
    DataLoaderKeys as DLK,
    PathKeys as PK
)
from configuration.parameter import (
    MODEL_BATCH_SIZE,
    TUNER_FOLDER,
    TUNE
)
from models.model_randomness import set_tf_seed


class TrainerTuner(TrainerInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tuner_dir = os.path.join(self.log_dir, TUNER_FOLDER)
        self.base_model = None
        self.best_hp = None
        self.best_model = None

    def search_for_hyper_parameter(self, tuning_data_paths: List[str], patient_names: List[str]):
        self.base_model, tuner = self.get_tuner(mirrored_strategy=self.mirrored_strategy)
        self.best_hp, self.best_model = self.search(tuner=tuner,
                                                    tuning_data_paths=tuning_data_paths,
                                                    patient_names=patient_names)

    def train_process(self, train_log_dir: str, datasets: tuple, class_weights: Dict[int, float], batch_path: str):
        # -------TRAINING---------
        if self.base_model or self.best_hp or self.best_model:
            raise TypeError("Execute 'search_for_hyper_parameter' before hyper model training!")

        history = self.base_model.fit(hp=self.best_hp,
                                      model=self.best_model,
                                      x=datasets[0],
                                      validation_data=datasets[1],
                                      verbose=2,
                                      epochs=self.config.CONFIG_TRAINER[TK.EPOCHS],
                                      batch_size=MODEL_BATCH_SIZE,
                                      callbacks=self.get_callbacks(train_log_dir=train_log_dir),
                                      use_multiprocessing=True,
                                      class_weight=class_weights,
                                      workers=int(os.cpu_count()))

        return self.best_model, history

    def restore_tuner(self, directory=''):
        if directory == '':
            directory = self.tuner_dir

        with open(os.path.join(directory, "model.pickle"), "rb") as handle:
            model = pickle.load(handle)

        with open(os.path.join(directory, "params.pickle"), "rb") as handle:
            params = pickle.load(handle)

        # params["objective"] = kt.Objective(**params["objective"])
        params["overwrite"] = False  # important, otherwise results could be overwritten and restoring will not work

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

    def search(self, tuner: kt.Tuner, tuning_data_paths: List[str], patient_names: List[str]) \
            -> tuple[kt.HyperParameters, keras.Model]:
        print("---- Get tuning datasets ----")
        tune_batch_path = os.path.join(self.config.CONFIG_PATHS[PK.BATCHED_PATH], TUNE)
        if not os.path.exists(tune_batch_path):
            os.makedirs(tune_batch_path)

        dataset_t = self.dataset.get_dataset(dataset_paths=tuning_data_paths,
                                             names=patient_names,
                                             labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                             batch_path=tune_batch_path,
                                             dataset_type="tuning")
        class_weights_t = self.get_class_weights(train_names=patient_names,
                                                 dataset_paths=tuning_data_paths)

        callbacks = [keras.callbacks.TensorBoard(self.tuner_dir),
                     get_early_stopping(
                         early_stopping_config=self.config.CONFIG_TRAINER[TK.CALLBACKS][TK.EARLY_STOPPING])]

        print("------ Start search tuning parameter ---------")
        # TODO split dataset in valid an train by validation split
        tuner.search(x=dataset_t,
                     epochs=self.config.CONFIG_TRAINER[TK.TUNER_EPOCHS],
                     callbacks=callbacks,
                     validation_split=1 - self.config.CONFIG_TRAINER[TK.SPLIT_FACTOR],
                     batch_size=self.config.CONFIG_TRAINER[TK.BATCH_SIZE],
                     class_weight=class_weights_t,
                     verbose=2)

        print("------ Finish search tuning parameter ---------")
        tuner.results_summary(num_trials=1)

        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]

        self.dataset.delete_batches(batch_path=tune_batch_path)

        return best_hp, best_model

    def get_tuner(self, mirrored_strategy=None) -> tuple[kt.HyperModel, kt.Tuner]:
        if self.config.CONFIG_TRAINER[TK.RESTORE]:
            base_model, params = self.restore_tuner()
        else:
            base_model = self.get_model()
            params = self.get_params()
            TrainerTuner.save_tuner_params(base_model, **params)
        tuner = self.config.CONFIG_TRAINER[TK.TUNER](base_model, **params, distribution_strategy=mirrored_strategy)

        if self.config.CONFIG_TRAINER[TK.RESTORE]:
            # in case of restoration model is not initialized and seed is not set, that's why error is thrown
            set_tf_seed()
            tuner.reload()

        return base_model, tuner

    def get_model(self) -> kt.HyperModel:
        base_model = self.config.CONFIG_TRAINER[TK.MODEL](
            input_shape=self.get_output_shape(),
            model_config=self.config.CONFIG_TRAINER[TK.MODEL_CONFIG],
            num_of_labels=len(self.config.CONFIG_DATALOADER[
                                  DLK.LABELS_TO_TRAIN]),
            custom_metrics=self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS],
            with_sample_weights=self.config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS])

        return base_model


if __name__ == '__main__':
    """from configuration.get_config import CVConfig
    import provider
    from configuration.keys import PathKeys as PK
    import numpy as np
    from glob import glob
    from pprint import pprint

    config = CVConfig()

    try:

        data_storage = provider.get_data_storage(typ=STORAGE_TYPE)
        log_dir = os.path.join(*config.CONFIG_PATHS[PK.LOGS_FOLDER], 'keras_tuner')

        trainer = TrainerTuner(config=config, data_storage=data_storage, model_name=log_dir,
                               leave_out_names=[], train_names=[],
                               valid_names=[])

        _, tuner = trainer.get_tuner()

        # print(tuner.results_summary())

        # shuffled = ['C:\\Users\\tkachenko\\Desktop\\HSI\\colon_for_debug\\keras_tuner\\shuffled\\shuffled_example.npz']
        shuffled = ['/work/mi186veva-MySpace/WHOLE_MainExperiment_7_smoothing_2d/run/shuffled/shuffled_example.npz']

        data = np.load(shuffled[0])

        callbacks = [keras.callbacks.TensorBoard(trainer.tuner_dir)]
        callbacks = trainer.add_early_stopping(callbacks)

        class_weigths = None
        if not config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS]:
            dat_files = glob(config.CONFIG_PATHS(PK.DATABASE_ROOT_FOLDER), '*.dat')
            trainer.train_names = [f.split(config.CONFIG_PATHS(PK.SYS_DELIMITER))[-1].split('_Spec')[0] for f in
                                   dat_files]
            class_weigths = trainer.get_class_weights(shuffled)

        tuner.search(x=data['X'],
                     y=data['y'],
                     epochs=config.CONFIG_TRAINER[TK.TUNER_EPOCHS],
                     # validation_data=valid_dataset,
                     class_weight=class_weigths,
                     callbacks=callbacks,
                     validation_split=1 - config.CONFIG_TRAINER[TK.SPLIT_FACTOR],
                     batch_size=config.CONFIG_TRAINER[TK.BATCH_SIZE],
                     # batch_size=100,
                     verbose=2)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models()[0]

        print('Best hyperparams')
        pprint(best_hps.values)
        print('Best model summary')
        print(best_model.summary())

        config.telegram.send_tg_message(f'Tuner succesfully finished')
    except Exception as e:
        config.telegram.send_tg_message(f'TUNER ERROR!!! {e}')"""

    '''best_model.fit(
        x=data['X'][::10],
        y=data['y'][::10],
        epochs=config.CONFIG_TRAINER[TK.TUNER_EPOCHS]
    )'''

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
