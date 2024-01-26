import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import inspect
from glob import glob

from data_utils.data_loaders.data_loader import DataLoader
from models.model_randomness import set_tf_seed

tf.random.set_seed(1)


class Predictor:

    def __init__(self, config):
        self.config = config
        set_tf_seed()

    def save_predictions(self, training_csv_path,
                         npz_folder,
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

                name = DataLoader(self.config).get_name(row[4], delimiter='/')
                print(f'We get checkpoint {checkpoint} for {model_path}')

                self.model = tf.keras.models.load_model(os.path.join(model_path,
                                                                     self.config.CONFIG_PATHS["CHECKPOINT_FOLDER"],
                                                                     checkpoint),
                                                        custom_objects=custom_objects)
                # predictor = Predictor(self.config, checkpoint, MODEL_FOLDER=model_path)
                predictions, gt, size = self.get_predictions_for_npz(os.path.join(npz_folder, name + ".npz"))

                results_dictionary.append({
                    'name': name,
                    'predictions': predictions,
                    'gt': gt,
                    'size': size,
                    'checkpoint': checkpoint
                })

        # saving of predictions
        np.save(os.path.join(predictions_saving_folder, predictions_npy_filename), results_dictionary)

    def get_predictions_for_npz(self, path):
        data = np.load(path)
        spectrum = data["X"]
        gt = data["y"]
        size = None
        if "size" in data:
            size = data["size"]

        # get only needed samples
        indexes = np.zeros(gt.shape).astype(bool)
        if not self.config.CONFIG_CV["USE_ALL_LABELS"]:
            for label in self.config.CONFIG_DATALOADER["LABELS_TO_TRAIN"]:
                indexes = indexes | (gt == label)
        else:
            indexes = np.ones(gt.shape).astype(bool)

        gt = gt[indexes]
        spectrum = spectrum[indexes]

        predictions = self.model.predict(spectrum, verbose=0)

        return predictions, gt, size

    def edit_model_path_if_local(self, model_path):
        if "LOCAL" in self.config.CONFIG_PATHS["MODE"]:
            model_path = model_path.split("hsi-experiments")[-1][1:]
            model_path = model_path.replace("/", "\\")

            current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            parent_dir = os.path.dirname(current_dir)
            model_path = os.path.join(parent_dir, model_path)
        return model_path

    def get_checkpoint(self, checkpoint, model_path=None):
        if type(checkpoint) == int:
            return f"cp-{checkpoint:04d}"
        if self.config.CONFIG_CV["GET_CHECKPOINT_FROM_EARLYSTOPPING"]:
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


if __name__ == "__main__":
    import configuration.get_config as configuration

    predictor_ = Predictor(configuration, f'cp-0020',
                           MODEL_FOLDER='/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_inception'
                                        '/3d_0_1_2_3')
