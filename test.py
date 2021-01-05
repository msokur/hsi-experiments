import tensorflow as tf 
import cv2
import numpy as np
import os
from tqdm import tqdm
from data_loader import *
from callbacks import CustomTensorboardCallback
from sklearn.metrics import confusion_matrix
import csv



class Tester():

    def __init__(self, LOGS_PATH, MODEL_NAME, CHECKPOINT, TEST_PATHS, SAVING_PATH):

        self.MODEL_NAME = MODEL_NAME
        MODEL_FOLDER = os.path.join(LOGS_PATH, MODEL_NAME)
        CHECKPOINTS_FOLDER_NAME = os.path.join(LOGS_PATH, MODEL_NAME, 'checkpoints')
        MODEL_PATH = os.path.join(CHECKPOINTS_FOLDER_NAME, CHECKPOINT)
        self.CHECKPOINT = CHECKPOINT

        self.TEST_PATHS = TEST_PATHS
        self.SAVING_PATH = SAVING_PATH

        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.tensorboard_callback = CustomTensorboardCallback(log_dir=MODEL_FOLDER) #for drawing function

        self.scaler = restore_scaler(MODEL_FOLDER)

    def count_metrics(self, gt, predictions, name, folder_name = ''):
        if folder_name == '':
            folder_name = self.SAVING_PATH

        conf_matrix = confusion_matrix(gt, predictions)

        tn = conf_matrix[0, 0]
        tp = conf_matrix[1, 1]
        fn = conf_matrix[1, 0]
        fp = conf_matrix[0, 1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print('name', name, ', sensitivity: ', sensitivity, ', specificity: ', specificity)

        with open(os.path.join(folder_name, 'stats.csv'),'a', newline='') as csvfile:
            fieldnames = ['name','sensitivity', 'specificity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'name':name, 'sensitivity':str(sensitivity), 'specificity':str(specificity)})

        return sensitivity, specificity

    def test_one_image(self, path_dat, path_image=None, save=False, show=True, test_all_spectra = False, save_stats = False, folder_name = ''):
        if folder_name == '':
            folder_name = self.SAVING_PATH

        gt_image, spectrum_data, gesund_indexes, ill_indexes, not_certain_indexes = get_data_for_showing(path_dat, "")
        indexes = gesund_indexes + ill_indexes + not_certain_indexes
        indexes = np.array(indexes)

        #scaler = restore_scaler(FOLDER_NAME)

        if  test_all_spectra:
            indexes = np.where(gt_image[:, :, 0] < 2055)
            indexes = np.array(indexes).T

        spectrum = self.scaler.transform(spectrum_data[indexes[:, 0], indexes[:, 1]])
        predictions = self.model.predict(np.expand_dims(spectrum, axis=-1))

        name = path_dat.split('\\')[-1].split('_S')[0]

        sensitivity = specificity = 0
        if save_stats and not test_all_spectra:
            gt = [0] * len(gesund_indexes) + [1] * len(ill_indexes)
            sensitivity, specificity = self.count_metrics(gt, np.rint(predictions), name, folder_name)

            global all_predictions
            global all_gt
            all_predictions += list(np.rint(predictions))
            all_gt += gt

        self.tensorboard_callback.gt_image = gt_image[..., ::-1]
        self.tensorboard_callback.spectrum = spectrum
        self.tensorboard_callback.indexes = indexes

        result_image = []
        if save or show:
            if path_image == None:
                result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=None)
            else:
                image = cv2.imread(path_image)
                result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=image)

            result_image = result_image[..., ::-1]

            if save:
                cv2.imwrite(os.path.join(folder_name, name + '.png'), result_image)

            if show:
                cv2.imshow('frame', result_image)
                cv2.waitKey(0)

        return sensitivity, specificity

    def test_ALL_images(self, save=True, show=False, test_all_spectra = False, save_stats = False):
        global all_predictions
        global all_gt

        for path_dir in self.TEST_PATHS:

            all_predictions = []
            all_gt = []
            name = self.MODEL_NAME + '_' + '_all_spectra_' + str(test_all_spectra) + '_' + self.CHECKPOINT + '_' + path_dir
            folder_name = os.path.join(self.SAVING_PATH, name)
            os.mkdir(folder_name)

            for path in tqdm(glob.glob(os.path.join(path_dir, '*.dat'))):
                with open(path, newline='') as filex:
                    filename=filex.name

                    self.test_one_image(filename, save=save, show=show, test_all_spectra=test_all_spectra, save_stats = save_stats, folder_name=folder_name)

            if save_stats:
                self.count_metrics(all_gt, all_predictions, 'GESAMT', folder_name)
                self.count_metrics(all_gt, all_predictions, 'GESAMT', folder_name)



if __name__ == "__main__":

    dat_name = r'test_test\2019_09_09_17_01_38_SpecCube.dat'

    #test_one_image(dat_name, path_image=dat_name + '_Mask JW Kolo.png', save=False, show=False, test_all_spectra=False, save_stats=True)
    tester = Tester('logs', 'inception_sample_weight_fixed_spectrum_dropout', 'cp-0050', ['data'], 'test')
    tester.test_ALL_images(test_all_spectra=False, save=True, show=False, save_stats=True)


