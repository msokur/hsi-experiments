import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from glob import glob

from configuration.keys import DataLoaderKeys as DLK, CrossValidationKeys as CVK, PathKeys as PK
from models.model_randomness import set_tf_seed
from provider import get_data_loader, get_data_storage
from configuration.parameter import (
    STORAGE_TYPE, MAX_SIZE_PER_SPEC, ORIGINAL_NAME
)

tf.random.set_seed(1)


class Predictor:

    def __init__(self, config):
        self.config = config
        self.data_storage = get_data_storage(typ=STORAGE_TYPE)
        set_tf_seed()

    def save_predictions(self, training_csv_path,
                         folder_with_npz,
                         predictions_saving_folder,
                         predictions_npy_filename,
                         checkpoint=None):
        """
            param rows of training_csv_path:
            0 - date
            1 - index
            2 - sensitivity
            3 - specificity
            4 - .dat path
            5 - model path
        """
        custom_objects = self.config.CONFIG_TRAINER["CUSTOM_OBJECTS_LOAD"]
        results_dictionary = []
        with open(training_csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                print(', '.join(row))

                model_path = self.edit_model_path_if_local(row[5])
                checkpoint = self.get_checkpoint(checkpoint, model_path=model_path)

                name = self.data_storage.get_name(row[4])
                print(f'We get checkpoint {checkpoint} for {model_path}')

                self.model = tf.keras.models.load_model(os.path.join(model_path,
                                                                     self.config.CONFIG_PATHS["CHECKPOINT_FOLDER"],
                                                                     checkpoint),
                                                        custom_objects=custom_objects)
                # predictor = Predictor(self.config, checkpoint, MODEL_FOLDER=model_path)
                predictions, gt, size, names = self.get_predictions_for_npz(os.path.join(folder_with_npz, name))

                results_dictionary.append({
                    'name': name,
                    'predictions': predictions,
                    'gt': gt,
                    ORIGINAL_NAME: names,
                    'size': size,
                    'checkpoint': checkpoint
                })

        # saving of predictions
        np.save(os.path.join(predictions_saving_folder, predictions_npy_filename), results_dictionary)

    def get_predictions_for_npz(self, path):
        data = self.data_storage.get_datas(data_path=path)
        spectrum = data["X"][...]
        gt = data["y"][...]
        size = None
        if "size" in data:
            size = data["size"][...]

        # get only needed samples
        indexes = np.zeros(gt.shape).astype(bool)
        if not self.config.CONFIG_CV["USE_ALL_LABELS"]:
            for label in self.config.CONFIG_DATALOADER["LABELS_TO_TRAIN"]:
                indexes = indexes | (gt == label)
        else:
            indexes = np.ones(gt.shape).astype(bool)

        gt = gt[indexes]
        spectrum = spectrum[indexes]
        prediction_steps, steps_width = self._get_spec_step(shape=spectrum.shape, dtype=spectrum.dtype)
        predictions_ = []
        start_step = 0
        for prediction_step in range(prediction_steps):
            if prediction_step == prediction_steps - 1:
                end_step = spectrum.shape[0]
            else:
                end_step = start_step + steps_width

            tf_spectrum = tf.convert_to_tensor(spectrum[start_step:end_step])
            predict_data = tf.data.Dataset.from_tensor_slices(tf_spectrum).batch(500)
            predictions_.append(self.model.predict(predict_data, verbose=0))
            start_step += steps_width

        predictions = np.concatenate(predictions_, axis=0)

        names = None
        if ORIGINAL_NAME in data:
            names = data[ORIGINAL_NAME][...][indexes]

        # TODO can be deleted in future, is for old datasets
        if "org_name" in data:
            names = data["org_name"][...][indexes]

        return predictions, gt, size, names

    def edit_model_path_if_local(self, model_path):
        if "LOCAL" in self.config.CONFIG_PATHS["MODE"]:
            model_folders = os.path.split(model_path)
            model_path = os.path.join(self.config.CONFIG_PATHS[PK.RESULTS_FOLDER], model_folders[-2], model_folders[-1])
        return model_path

    def get_checkpoint(self, checkpoint, model_path=None):
        if type(checkpoint) == int:
            return f"cp-{checkpoint:04d}"
        if self.config.CONFIG_CV[CVK.GET_CHECKPOINT_FROM_EARLYSTOPPING]:
            return self.get_best_checkpoint_from_csv(model_path)
        if checkpoint is None:
            return f"cp-{self.config.CONFIG_TRAINER['EPOCHS']:04d}"

        return checkpoint

    def get_best_checkpoint_from_csv(self, model_path):
        if model_path is None:
            raise ValueError('Please specify model path!')

        checkpoints_paths = sorted(glob(os.path.join(model_path,
                                                     self.config.CONFIG_PATHS["CHECKPOINT_FOLDER"], "*"
                                                     + self.config.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])))

        best_checkpoint_path = checkpoints_paths[-1]
        return best_checkpoint_path.split(self.config.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"])[-2]

    @staticmethod
    def _get_spec_step(shape: tuple, dtype: np.dtype, max_size: float = MAX_SIZE_PER_SPEC):
        if dtype == np.float64:
            byte = 8
        else:
            byte = 4

        size = (np.prod(shape) * byte) / (1024 ** 3)

        if size <= max_size:
            # return only one step and the size of the
            return 1, shape[0]
        else:
            # calculate the needed steps
            steps = int(-(-size // max_size))
            # calculate the size for every step
            step_size = int(-(-shape[0] // steps))

            return steps, step_size


if __name__ == "__main__":
    import configuration.get_config as configuration

    predictor_ = Predictor(configuration)
