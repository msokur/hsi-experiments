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

    '''
    there are two variants
    1. give LOGS_PATH and MODEL_PATH separatly
    2. give MODEL_FOLDER
    '''
    def __init__(self, CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER=''):

        if MODEL_NAME != '':
            self.MODEL_NAME = MODEL_NAME
        else:
            self.MODEL_NAME = MODEL_FOLDER.split('\\')[-1] #here can be problem

        if MODEL_FOLDER == '':
            MODEL_FOLDER = os.path.join(LOGS_PATH, self.MODEL_NAME)

        #CHECKPOINTS_FOLDER_NAME = os.path.join(LOGS_PATH, self.MODEL_NAME, 'checkpoints')
        CHECKPOINTS_FOLDER_NAME = os.path.join(MODEL_FOLDER, 'checkpoints')
        MODEL_PATH = os.path.join(CHECKPOINTS_FOLDER_NAME, CHECKPOINT)
        self.CHECKPOINT = CHECKPOINT

        self.TEST_PATHS = TEST_PATHS
        self.SAVING_PATH = SAVING_PATH

        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.tensorboard_callback = CustomTensorboardCallback(log_dir=MODEL_FOLDER) #for drawing function

        self.scaler = restore_scaler(MODEL_FOLDER)

        self.all_predictions = []
        self.all_predictions_raw = []
        self.all_gt = []

    @staticmethod
    def count_metrics(gt, predictions, name, folder_name='', save_stats=True):
        if save_stats and folder_name == '':
            folder_name = self.SAVING_PATH

        conf_matrix = confusion_matrix(gt, predictions)

        tn = conf_matrix[0, 0]
        tp = conf_matrix[1, 1]
        fn = conf_matrix[1, 0]
        fp = conf_matrix[0, 1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print('name', name, ', sensitivity: ', sensitivity, ', specificity: ', specificity)

        if save_stats:
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
        if not test_all_spectra:
            gt = [0] * len(gesund_indexes) + [1] * len(ill_indexes)
            sensitivity, specificity = self.count_metrics(gt, np.rint(predictions), name, folder_name, save_stats)

            self.all_predictions += list(np.rint(predictions))
            self.all_predictions_raw += list(predictions)
            self.all_gt += gt

        self.tensorboard_callback.gt_image = gt_image#[..., ::-1]
        self.tensorboard_callback.spectrum = spectrum
        self.tensorboard_callback.indexes = indexes

        result_image = []
        if save or show:
            if path_image == None:
                result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=None)
            else:
                image = cv2.imread(path_image)
                result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=image)

            #result_image = result_image[..., ::-1]

            if save:
                cv2.imwrite(os.path.join(folder_name, name + '.png'), result_image)

            if show:
                cv2.imshow('frame', result_image)
                cv2.waitKey(0)

        return sensitivity, specificity

    def test_ALL_images(self, save=True, show=False, test_all_spectra = False, save_stats = False):

        for path_dir in self.TEST_PATHS:

            self.all_predictions = []
            self.all_predictions_raw = []
            self.all_gt = []
            name = self.MODEL_NAME + '_' + '_all_spectra_' + str(test_all_spectra) + '_' + self.CHECKPOINT + '_' + path_dir
            folder_name = os.path.join(self.SAVING_PATH, name)
            os.mkdir(folder_name)

            for path in tqdm(glob.glob(os.path.join(path_dir, '*.dat'))):
                with open(path, newline='') as filex:
                    filename=filex.name

                    self.test_one_image(filename, save=save, show=show, test_all_spectra=test_all_spectra, save_stats = save_stats, folder_name=folder_name)

            if save_stats:
                self.count_metrics(self.all_gt, self.all_predictions, 'GESAMT', folder_name)
                self.count_metrics(self.all_gt, self.all_predictions, 'GESAMT', folder_name)



if __name__ == "__main__":

    #dat_names = [r'data\2019_07_12_11_15_49_SpecCube.dat',r'data\2019_07_17_15_38_14_SpecCube.dat', r'data\2019_07_25_11_56_38_SpecCube.dat', r'data\2019_08_09_12_17_55_SpecCube.dat' ]

    #test_one_image(dat_name, path_image=dat_name + '_Mask JW Kolo.png', save=False, show=False, test_all_spectra=False, save_stats=True)

    #(self, CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER=''):

    tester = Tester('cp-0250', ['data'], 'test', MODEL_FOLDER='logs\lstm')

    #for dat_name in dat_names:
    #    tester.test_one_image(dat_name, path_image=dat_name + '_Mask JW Kolo.png', save=False, show=False, test_all_spectra=False, save_stats=True, folder_name='logs\inception_l2_norm\inception_l2_norm_0_1_2_3')
    tester.test_ALL_images(test_all_spectra=False, save=True, show=False, save_stats=True)


