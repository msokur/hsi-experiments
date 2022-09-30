import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from glob import glob
import os
import inspect

import config
from data_utils.data_loaders.data_loader_base import DataLoader
from evaluation.metrics import Metrics


class Predictor:
    """
    there are two variants:
    1. give LOGS_PATH and MODEL_PATH separately
    2. give MODEL_FOLDER
    """

    def __init__(self, CHECKPOINT, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER='', custom_objects=config.CUSTOM_OBJECTS):

        if MODEL_NAME != '':
            self.MODEL_NAME = MODEL_NAME
        else:
            self.MODEL_NAME = MODEL_FOLDER.split(config.SYSTEM_PATHS_DELIMITER)[-1]  # here can be problem

        if MODEL_FOLDER == '':
            MODEL_FOLDER = os.path.join(LOGS_PATH, self.MODEL_NAME)

        CHECKPOINTS_FOLDER_NAME = os.path.join(MODEL_FOLDER, 'checkpoints')
        MODEL_PATH = os.path.join(CHECKPOINTS_FOLDER_NAME, CHECKPOINT)
        self.CHECKPOINT = CHECKPOINT

        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

        print('--------------------PARAMS----------------------')
        print(', \n'.join("%s: %s" % item for item in vars(self).items()))
        print('------------------------------------------------')

    def get_predictions_for_npz(self, path):
        data = np.load(path)
        spectrum = data['X']
        gt = data['y']
        size = None
        if 'size' in data:
            size = data['size']

        # get only needed samples
        indexes = np.zeros(gt.shape).astype(bool)
        if not config.USE_ALL_LABELS:
            for label in config.LABELS_OF_CLASSES_TO_TRAIN:
                indexes = indexes | (gt == label)
        else:
            indexes = np.ones(gt.shape).astype(bool)
        if config.WITH_BACKGROUND_EXTRACTION:
            gt = gt[indexes & data['bg_mask']]
            spectrum = spectrum[indexes & data['bg_mask']]

        gt = gt[indexes]
        spectrum = spectrum[indexes]

        predictions = self.model.predict(spectrum)

        return predictions, gt, size

    @staticmethod
    def edit_model_path_if_local(model_path):
        if 'LOCAL' in config.MODE:
            model_path = model_path.split('hsi-experiments')[-1][1:]
            model_path = model_path.replace('/', '\\')

            current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            parent_dir = os.path.dirname(current_dir)
            model_path = os.path.join(parent_dir, model_path)
        return model_path

    @staticmethod
    def get_best_checkpoint_from_csv(model_path):
        checkpoints_paths = sorted(glob(os.path.join(model_path,
                                                     'checkpoints' + config.SYSTEM_PATHS_DELIMITER + '*'
                                                     + config.SYSTEM_PATHS_DELIMITER)))
        best_checkpoint_path = checkpoints_paths[-1]
        return best_checkpoint_path.split(config.SYSTEM_PATHS_DELIMITER)[-2]

    @staticmethod
    def get_checkpoint(checkpoint, model_path):
        if checkpoint is None:
            checkpoint = f'cp-{config.EPOCHS:04d}'

        if config.CV_GET_CHECKPOINT_FROM_VALID:
            return Predictor.get_best_checkpoint_from_csv(model_path)
        else:
            return checkpoint

    @staticmethod
    def save_predictions(training_csv_path,
                         npz_folder,
                         predictions_saving_folder,
                         predictions_npy_filename,
                         checkpoint=None,
                         save_roc_auc_curve=False):
        """
            param rows of training_csv_path:
            0 - date
            1 - index
            2 - sensitivity
            3 - specificity
            4 - .dat path
            5 - model path
        """

        all_predictions_raw, all_gt = [], []
        results_dictionary = []

        with open(training_csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                print(', '.join(row))

                model_path = Predictor.edit_model_path_if_local(row[5])

                checkpoint = Predictor.get_checkpoint(checkpoint, model_path)
                name = DataLoader.get_name_easy(row[4], delimiter='/')
                print(f'We get checkpoint {checkpoint} for {model_path}')

                predictor = Predictor(checkpoint, MODEL_FOLDER=model_path)
                predictions, gt, size = predictor.get_predictions_for_npz(os.path.join(npz_folder, name + ".npz"))

                results_dictionary.append({
                    'name': name,
                    'predictions': predictions,
                    'gt': gt,
                    'size': size
                })

                if save_roc_auc_curve:
                    all_predictions_raw += list(predictions)
                    all_gt += list(gt)

        # saving of predictions
        np.save(os.path.join(predictions_saving_folder, predictions_npy_filename), results_dictionary)

        # roc auc part (for all predictions together as one array)
        if save_roc_auc_curve:
            metr = Metrics()
            metr.save_roc_curves(all_gt,
                                 all_predictions_raw, "All predictions together", predictions_saving_folder)


if __name__ == "__main__":
    predictor_ = Predictor(f'cp-0020',
                           MODEL_FOLDER='/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_inception'
                                        '/3d_0_1_2_3')
