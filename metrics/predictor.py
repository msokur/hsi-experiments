import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import datetime

import config


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

        return predictions, gt


if __name__ == "__main__":

    tester = Predictor(f'cp-0020', [config.RAW_NPZ_PATH], '',
                       MODEL_FOLDER='/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_inception/3d_0_1_2_3')
