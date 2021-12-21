import sys
sys.path.insert(0, 'utils')
sys.path.insert(1, 'data_utils')

import config
import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
import callbacks
from sklearn.metrics import confusion_matrix, f1_score
import csv
from sklearn import preprocessing
import inspect
import glob

class Tester():

    '''
    there are two variants
    1. give LOGS_PATH and MODEL_PATH separatly
    2. give MODEL_FOLDER
    '''
    def __init__(self, CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER='', custom_objects=config.CUSTOM_OBJECTS):

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

        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects) #{'f1_m':train.f1_m}
        self.tensorboard_callback = callbacks.CustomTensorboardCallback(log_dir=MODEL_FOLDER) #for drawing function

        #self.scaler = restore_scaler(MODEL_FOLDER)
        self.scaler = preprocessing.Normalizer() #TODO return normal scaler

        self.all_predictions = []
        self.all_predictions_raw = []
        self.all_gt = []
        
        print('--------------------PARAMS----------------------')
        print(', \n'.join("%s: %s" % item for item in vars(self).items()))
        print('------------------------------------------------')

    @staticmethod
    def count_metrics(gt, predictions, name, folder_name='', save_stats=True, return_dice=False):
        print('--------------------method count_metrics params----------------------')
        signature = inspect.signature(Tester.count_metrics)
        for param in signature.parameters.values():
            print(param)
        print('------------------------------------------------')
        
        if save_stats and folder_name == '':
            folder_name = self.SAVING_PATH
            
        if len(predictions.shape) > 1:
            predictions = np.reshape(predictions, predictions.shape[0])
       
        predictions = predictions.astype(np.int)

        conf_matrix = confusion_matrix(gt, predictions, labels=[0,1])

        tn = conf_matrix[0, 0]
        tp = conf_matrix[1, 1]
        fn = conf_matrix[1, 0]
        fp = conf_matrix[0, 1]

        sensitivity = tp / (tp + fn) #recall
        specificity = tn / (tn + fp)

        F1 = f1_score(gt, predictions)#2 * precision * sensitivity / (precision + sensitivity) #DICE score

        print('name', name, ', sensitivity: ', sensitivity, ', specificity: ', specificity, ', F1-score(DICE): ', F1)

        if save_stats:
            with open(os.path.join(folder_name, 'stats.csv'),'a', newline='') as csvfile:
                fieldnames = ['name','sensitivity', 'specificity', 'F1']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'name':name,
                                 'sensitivity':str(sensitivity),
                                 'specificity':str(specificity),
                                 'F1': F1})

        if return_dice:
            return sensitivity, specificity, F1

        return sensitivity, specificity

    def test_one_image(self, path_dat, path_image=None, save=False, show=True, test_all_spectra=False, save_stats=False,
                       folder_name='', grayscale_result=False, return_dice=False, test_batch=False, spectrum_shift=0): #TODO remove test_batch
        
        print('--------------------method test_one_image params----------------------')
        signature = inspect.signature(self.test_one_image)
        print('path_dat: ', path_dat)
        for param in signature.parameters.values():
            print(param)
        print('------------------------------------------------')
        
        print('test_batch', test_batch)
        
        if folder_name == '':
            folder_name = self.SAVING_PATH
       
        #scaler = restore_scaler(FOLDER_NAME)        

        
        if test_batch:
            data = np.load(path_dat)
            spectrum = data['X']  #test batch
            gt_image = cv2.imread(path_image)
            print('test_batch shape', spectrum.shape)
        else:
            gt_image, spectrum_data, gesund_indexes, ill_indexes, not_certain_indexes = data_loader.get_data_for_showing(path_dat, "")
            indexes = gesund_indexes + ill_indexes + not_certain_indexes
            indexes = np.array(indexes)
            
            if test_all_spectra:
                indexes = np.where(gt_image[:, :, 0] < 2055)
                indexes = np.array(indexes).T
            
            spectrum = spectrum_data[indexes[:, 0], indexes[:, 1]]
            
            
        if spectrum_shift != 0:
            spectrum = spectrum[..., :spectrum_shift]
        #spectrum = self.scaler.transform(spectrum) #test_batch 
            
        if test_batch:
            gt = data['y']  #test batch
            indx_ = ((gt == 0) | (gt == 1))
            if config.WITH_BACKGROUND_EXTRACTION:
                gt = gt[indx_ & data['bg_mask']]
                spectrum = spectrum[indx_ & data['bg_mask']]
            elif not config.NOT_CERTAIN_FLAG:
                gt = gt[indx_]
                spectrum = spectrum[indx_]

        else:
            gt = [0] * len(gesund_indexes) + [1] * len(ill_indexes)
            
            
        #predictions = self.model.predict(np.expand_dims(spectrum, axis=-1))
        predictions = self.model.predict(spectrum)
        print('dsd', gt.shape, predictions.shape)

        
        name = path_dat.split('\\')[-1].split('_S')[0]

        sensitivity = specificity = 0
        if not test_all_spectra:
            
            if not config.NOT_CERTAIN_FLAG:
                sensitivity, specificity = self.count_metrics(gt, np.rint(predictions), name, folder_name, save_stats, return_dice=return_dice)

            self.all_predictions += list(np.rint(predictions))
            self.all_predictions_raw += list(predictions)
            #
            if test_batch:
                self.all_gt += list(gt)   #test batch
            else:
                self.all_gt += gt

        #self.tensorboard_callback.gt_image = gt_image#[..., ::-1]   ОБЯЗАТЕЛЬНО ЭТО ВЕРНУТЬ!!!!!
        #self.tensorboard_callback.spectrum = spectrum
        #self.tensorboard_callback.indexes = indexes

        result_image = []
        if save or show:
            if path_image == None:
                result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=None,
                                                                                    grayscale_result=grayscale_result)
            else:
                image = cv2.imread(path_image)
                result_image = self.tensorboard_callback.draw_predictions_on_images(predictions, image=image,
                                                                                    grayscale_result=grayscale_result)

            #result_image = result_image[..., ::-1]

            if save:
                cv2.imwrite(os.path.join(folder_name, name + '.png'), result_image)

            if show:
                cv2.imshow('frame', result_image)
                cv2.waitKey(0)

        return sensitivity, specificity

    def test_ALL_images(self, save=True,
                        show=False,
                        test_all_spectra=False,
                        save_stats=False,
                        grayscale_result=False,
                        include_indexes=None,
                        exclude_indexes=None,
                        return_dice=False):

        for path_dir in self.TEST_PATHS:

            self.all_predictions = []
            self.all_predictions_raw = []
            self.all_gt = []
            p = path_dir
            p = p.replace("\\", "-")
            
            folder_name=""
            if save_stats or save:
                name = self.MODEL_NAME + '_' + '_all_spectra_' + str(test_all_spectra) + '_' + self.CHECKPOINT + '_' + p + '_gray_' + str(grayscale_result)
                folder_name = os.path.join(self.SAVING_PATH, name)
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)

            for i, path in tqdm(enumerate(glob.glob(os.path.join(path_dir, '*.dat')))):

                if i in include_indexes if include_indexes is not None else True:
                    if not i in exclude_indexes if exclude_indexes is not None else True:
                        with open(path, newline='') as filex:
                            filename=filex.name

                            self.test_one_image(filename,
                                                save=save,
                                                show=show,
                                                test_all_spectra=test_all_spectra,
                                                save_stats = save_stats,
                                                folder_name=folder_name,
                                                grayscale_result=grayscale_result)

            if save_stats:
                self.count_metrics(self.all_gt, self.all_predictions, 'GESAMT', folder_name, return_dice=return_dice)
                self.count_metrics(self.all_gt, self.all_predictions, 'GESAMT', folder_name, return_dice=return_dice)



if __name__ == "__main__":
    
    tester = Tester( f'cp-0020', config.DATA_PATHS, '', MODEL_FOLDER='/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_inception/3d_0_1_2_3')

    #tester.test_ALL_images(save=False, show=False, test_all_spectra=False, save_stats=False)
    
    
    path = '/work/users/mi186veva/data/2019_09_04_12_43_40_SpecCube.dat'
    paths = glob.glob('/work/users/mi186veva/data_preprocessed/raw_3d/*.npz')
    for path in ['2019_07_15_11_33_28_', '2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_']:#, '/work/users/mi186veva/data/2020_05_28_15_20_27_SpecCube.dat', '/work/users/mi186veva/data/2019_07_12_11_15_49_SpecCube.dat', '/work/users/mi186veva/data/2020_05_15_12_43_58_SpecCube.dat']:
        path = '/work/users/mi186veva/data_preprocessed/raw_3d/' + path + '.npz'
        sensitivity, specificity = tester.test_one_image(path,
                            path_image=path + '_Mask JW Kolo.png',
                            save=False,
                            show=False,
                            test_all_spectra=False,
                            save_stats=False,
                            folder_name=config.MODEL_NAME,
                            test_batch=True,
                            spectrum_shift=0)
    print(sensitivity, specificity)

    #dat_names = [r'data\2019_07_12_11_15_49_SpecCube.dat',r'data\2019_07_17_15_38_14_SpecCube.dat', r'data\2019_07_25_11_56_38_SpecCube.dat', r'data\2019_08_09_12_17_55_SpecCube.dat' ]

    #test_one_image(dat_name, path_image=dat_name + '_Mask JW Kolo.png', save=False, show=False, test_all_spectra=False, save_stats=True)

    #(self, CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER=''):

    '''rg = np.linspace(100, 200, 5).astype(int)
    checkpoints = [f'cp-{i:04d}' for i in rg]
    print(checkpoints)
    #f'cp-{config.EPOCHS:04d}'
    for checkpoint in checkpoints:
        tester = Tester(checkpoint, [r'data'], 'test', MODEL_FOLDER='logs\lstm_inception_8_without_35')
        tester.test_one_image('data/2019_08_28_14_00_34_SpecCube.dat', save=False, show=False, test_all_spectra=False,
                              save_stats=True, folder_name='test/lstm_inception_8_without_35_test_checkpoints')'''

    #for dat_name in dat_names:
    #    tester.test_one_image(dat_name, path_image=dat_name + '_Mask JW Kolo.png', save=False, show=False, test_all_spectra=False, save_stats=True, folder_name='logs\inception_l2_norm\inception_l2_norm_0_1_2_3')
    #tester.test_ALL_images(test_all_spectra=False, save=False, show=False, save_stats=True, grayscale_result=False, include_indexes=[32,33,34])




