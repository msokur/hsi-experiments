import tensorflow as tf
import os
from sklearn import preprocessing
import abc
from tqdm import tqdm
from glob import glob

import config
import callbacks


class Tester():
    """
    there are two variants
    1. give LOGS_PATH and MODEL_PATH separately
    2. give MODEL_FOLDER
    """

    def __init__(self, CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER='',
                 custom_objects=config.CUSTOM_OBJECTS):

        if MODEL_NAME != '':
            self.MODEL_NAME = MODEL_NAME
        else:
            self.MODEL_NAME = MODEL_FOLDER.split(config.SYSTEM_PATHS_DELIMITER)[-1]  # here can be problem

        if MODEL_FOLDER == '':
            MODEL_FOLDER = os.path.join(LOGS_PATH, self.MODEL_NAME)

        # CHECKPOINTS_FOLDER_NAME = os.path.join(LOGS_PATH, self.MODEL_NAME, 'checkpoints')
        CHECKPOINTS_FOLDER_NAME = os.path.join(MODEL_FOLDER, 'checkpoints')
        MODEL_PATH = os.path.join(CHECKPOINTS_FOLDER_NAME, CHECKPOINT)
        self.CHECKPOINT = CHECKPOINT

        self.TEST_PATHS = TEST_PATHS
        self.SAVING_PATH = SAVING_PATH

        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)  # {'f1_m':train.f1_m}
        self.tensorboard_callback = callbacks.CustomTensorboardCallback(log_dir=MODEL_FOLDER)  # for drawing function

        # self.scaler = restore_scaler(MODEL_FOLDER)
        self.scaler = preprocessing.Normalizer()  # TODO return normal scaler

        self.all_predictions = []
        self.all_predictions_raw = []
        self.all_gt = []

        print('--------------------PARAMS----------------------')
        print(', \n'.join("%s: %s" % item for item in vars(self).items()))
        print('------------------------------------------------')

    @abc.abstractmethod
    def count_metrics(self, gt, predictions, name):
        pass

    @abc.abstractmethod
    def test_one_image(self, path, path_image=None, save=False, show=True, test_all_spectra=False,
                       saving_path='', grayscale_result=False,
                       spectrum_shift=0):
        pass

    def test_ALL_images(self, save=True,
                        show=False,
                        test_all_spectra=False,
                        save_stats=False,
                        grayscale_result=False,
                        include_indexes=None,
                        exclude_indexes=None):

        for path_dir in self.TEST_PATHS:

            self.all_predictions = []
            self.all_predictions_raw = []
            self.all_gt = []
            p = path_dir
            p = p.replace("\\", "-")

            folder_name = ""
            if save_stats or save:
                name = self.MODEL_NAME + '_' + '_all_spectra_' + str(
                    test_all_spectra) + '_' + self.CHECKPOINT + '_' + p + '_gray_' + str(grayscale_result)
                folder_name = os.path.join(self.SAVING_PATH, name)
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)

            for i, path in tqdm(enumerate(glob(os.path.join(path_dir, '*' + config.FILE_EXTENSION)))):
                if i in include_indexes if include_indexes is not None else True:
                    if i not in exclude_indexes if exclude_indexes is not None else True:
                        with open(path, newline='') as filex:
                            filename = filex.name

                            self.test_one_image(filename, save=save, show=show, test_all_spectra=test_all_spectra,
                                                saving_path=folder_name, grayscale_result=grayscale_result)

            if save_stats:
                self.count_metrics(self.all_gt, self.all_predictions, 'TOTAL', folder_name)
                self.count_metrics(self.all_gt, self.all_predictions, 'TOTAL', folder_name)

    def save_or_show_result_prediction_on_image(self, predictions,
                                                path_image,
                                                saving_path,
                                                name,
                                                grayscale_result=False,
                                                save=False, show=False):

        if save or show:
            import cv2
            # self.tensorboard_callback.gt_image = gt_image#[..., ::-1]   ОБЯЗАТЕЛЬНО ЭТО ВЕРНУТЬ!!!!!
            # self.tensorboard_callback.spectrum = spectrum
            # self.tensorboard_callback.indexes = indexes

            if path_image is not None:
                image = cv2.imread(path_image)
            result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=image,
                                                                                grayscale_result=grayscale_result)

            # result_image = result_image[..., ::-1]

            if save:
                cv2.imwrite(os.path.join(saving_path, name + '.png'), result_image)

            if show:
                cv2.imshow('frame', result_image)
                cv2.waitKey(0)
