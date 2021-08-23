from tensorflow import keras
import tensorflow as tf
import cv2
import data_loader #import get_data_for_showing, restore_scaler
import config
import numpy as np
from tqdm import tqdm
import os
import math
import glob
import datetime
import test
from sklearn import preprocessing

class CustomTensorboardCallback(keras.callbacks.TensorBoard):

    def __init__(self, except_indexes=[], **kwargs):
        print(except_indexes)
        
        super(CustomTensorboardCallback, self).__init__(**kwargs)
        
        self.test_name = '2019_09_04_12_43_40_SpecCube.dat'
        gt_image, spectrum_data, gesund_indexes, ill_indexes, not_certain_indexes = data_loader.get_data_for_showing(self.test_name, config.DATA_PATHS[0])
        indexes = gesund_indexes + ill_indexes 
        indexes = np.array(indexes)

        #scaler = data_loader.restore_scaler(self.log_dir) #TODO return normal scaling
        self.scaler =  preprocessing.Normalizer()
        #spectrum = spectrum_data[indexes[:, 0], indexes[:, 1]]
        spectrum = self.scaler.transform(spectrum_data[indexes[:, 0], indexes[:, 1]])
        self.gt_image = gt_image
        self.gt = [0] * len(gesund_indexes) + [1] * len(ill_indexes)
        self.spectrum = spectrum
        self.indexes = indexes

        self.are_excepted = False
        if len(except_indexes) > 0:
            self.are_excepted = True
            self.except_indexes = except_indexes
            self.get_spectrum_of_excluded_patients()
            

    def get_spectrum_of_excluded_patients(self):
        self.excepted_spectrums = []
        self.excepted_gt = []
        
        for except_name in self.except_indexes:
            path = glob.glob(os.path.join(config.RAW_NPY_PATH, except_name + '*'))
                        
            if len(path) == 0:
                print(f'WARNING! For except_name {except_name} no raw_paths were found')
            else:
                data = np.load(path[0])
                not_not_certain_indexes = np.flatnonzero(data['y'] != 2)
                self.excepted_spectrums.append(self.scaler.transform(data['X'][not_not_certain_indexes, :-1]))
                self.excepted_gt.append(data['y'][not_not_certain_indexes])
        
        
        self.excepted_spectrums = np.array(self.excepted_spectrums) 
        self.excepted_gt = np.array(self.excepted_gt)
                   


    def draw_predictions_on_images(self, predictions, image=None, grayscale_result=False):
        gt_image = self.gt_image.copy()
        if grayscale_result:
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY).astype(float)
        indexes = self.indexes

        if image is None:
            image = gt_image.copy()
        else:
            shapes = gt_image.shape
            border = int((image.shape[1] - shapes[1]) / 2)
            image = image[image.shape[0] - shapes[0] - border:image.shape[0] - border, border:image.shape[1] - border, :]
        
        
        print('Writing summary image ...')
        is_tf = True
        for counter, value in enumerate(predictions):
            if not grayscale_result:
                key = 0
                if type(value) == np.ndarray:
                    key = round(value[0])
                    is_tf = False
                else:
                    key = tf.round(value)

                if key == 0:
                    image[indexes[counter, 0], indexes[counter, 1]] = [255, 0, 0]
                    #image.itemset((indexes[counter, 0], indexes[counter, 1]), [255, 0, 0])
                elif key == 1:
                    image[indexes[counter, 0], indexes[counter, 1]] = [0, 255, 255]
                    #image.itemset((indexes[counter, 0], indexes[counter, 1]), [0, 255, 255])
                else:
                    image[indexes[counter, 0], indexes[counter, 1]] = [0, 0, 255]
                    #image.itemset((indexes[counter, 0], indexes[counter, 1]), [0, 0, 255])
            else:
                key = 0
                if type(value) == np.ndarray:
                    key = value[0]
                    is_tf = False
                else:
                    key = value

                image = image / 255.
                print(image.shape)
                print(key)
                print(max(image), min(image))

                image[indexes[counter, 0], indexes[counter, 1]] = key

        image = np.array(list(image) + list(gt_image))
        cv2.imwrite('test.png', image[... , ::-1])     #TODO delete
        if is_tf:
            image = image[..., ::-1]
        return image
    
    def __write_valid_scalar(self, scalar_name, scalar_value, epoch):
        if 'val' in self._writers:
            with self._writers['val'].as_default():
                tf.summary.scalar(scalar_name, data=scalar_value, step=epoch)
                #tf.summary.scalar('epoch_specificity', data=logs['val_tn'] / (logs['val_tn'] + logs['val_fp']), step=epoch)
        if 'validation' in self._writers:
            with self._writers['validation'].as_default():
                tf.summary.scalar(scalar_name, data=scalar_value, step=epoch)
                #tf.summary.scalar('epoch_specificity', data=logs['val_tn'] / (logs['val_tn'] + logs['val_fp']), step=epoch)

    def on_epoch_end(self, epoch, logs=None):

        super(CustomTensorboardCallback, self).on_epoch_end(epoch, logs)
        
        print('{0}, epoch {1} is ended'.format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), epoch))

        with self._writers['train'].as_default():
            if epoch % config.CHECKPOINT_WRITING_STEP == 0:
                if config.WRITE_IMAGES:
                    gt_image = self.gt_image
                    spectrum = self.spectrum

                    predictions = self.model.predict(np.expand_dims(spectrum, axis=-1))   # TODO don't forget to remove expand_dims (or better learn how to get the shape of an input layer)
                    image = tf.py_function(self.draw_predictions_on_images, [predictions], [tf.uint8])
                    tf.summary.image('image', image, step=epoch)
            
            tf.summary.scalar('epoch_specificity', data=logs['tn'] / (logs['tn'] + logs['fp']), step=epoch)
        
        self.__write_valid_scalar('epoch_specificity', logs['val_tn'] / (logs['val_tn'] + logs['val_fp']), epoch)
        
        if self.are_excepted:
            for name, exc, gt in zip(self.except_indexes, self.excepted_spectrums, self.excepted_gt):
                predictions = self.model.predict(exc[:, :-1])
                sensitivity, specificity, f1 = test.Tester.count_metrics(np.rint(gt), np.rint(predictions), "", "", False, return_dice=True)
                
                self.__write_valid_scalar('test_'+name+'_sensitivity', sensitivity, epoch)
                self.__write_valid_scalar('test_'+name+'_specificity', specificity, epoch)
                self.__write_valid_scalar('test_'+name+'_f1', f1, epoch)
        
                print(f'-------Epoch validation: {name} sensitivity:{sensitivity} specificity:{specificity} -----------')
            
        '''print(self.__dict__)
        for key, value in logs.items():
            print (key, value)'''