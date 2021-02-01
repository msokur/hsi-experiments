from tensorflow import keras
import tensorflow as tf
import cv2
from data_loader import get_data_for_showing, restore_scaler
import config
import numpy as np
from tqdm import tqdm
import os
import math


class CustomTensorboardCallback(keras.callbacks.TensorBoard):

    def __init__(self, **kwargs):
        
        super(CustomTensorboardCallback, self).__init__(**kwargs)

        gt_image, spectrum_data, gesund_indexes, ill_indexes, not_certain_indexes = get_data_for_showing('2019_07_12_11_15_49_SpecCube.dat', config.DATA_PATHS[0])
        indexes = gesund_indexes + ill_indexes + not_certain_indexes
        indexes = np.array(indexes)

        scaler = restore_scaler(self.log_dir)
        spectrum = scaler.transform(spectrum_data[indexes[:, 1], indexes[:, 0]])
        self.gt_image = gt_image
        self.spectrum = spectrum
        self.indexes = indexes

    def draw_predictions_on_images(self, predictions, image=None):
        gt_image = self.gt_image
        indexes = self.indexes

        if image is None:
            image = self.gt_image.copy()
        else:
            shapes = gt_image.shape
            border = int((image.shape[1] - shapes[1]) / 2)
            image = image[image.shape[0] - shapes[0] - border:image.shape[0] - border, border:image.shape[1] - border, :]
        
        
        print('Writing summary image ...')
        is_tf = True
        for counter, value in tqdm(enumerate(predictions)):
            key = 0
            if type(value) == np.ndarray:
                key = round(value[0])
                is_tf = False
            else:
                key = tf.round(value)

            if key == 0:
                #image[indexes[gt_image.shape[0] - counter - 1, 0], indexes[counter, 1]] = [0, 0, 255]#[255, 0, 0]
                #image[indexes[counter, 0], indexes[counter, 1]] = [0, 0, 255]
                image[indexes[counter, 0], indexes[counter, 1]] = [255, 0, 0]
            elif key == 1:
                #image[indexes[counter, 0], indexes[counter, 1]] = [255, 255, 0]
                image[indexes[counter, 0], indexes[counter, 1]] = [0, 255, 255]
            else:
                #image[indexes[counter, 0], indexes[counter, 1]] = [255, 0, 0]
                image[indexes[counter, 0], indexes[counter, 1]] = [0, 0, 255]

        image = np.array(list(image) + list(gt_image))
        cv2.imwrite('test.png', image[... , ::-1])     #TODO delete
        if is_tf:
            image = image[..., ::-1]
        return image

    def on_epoch_end(self, epoch, logs=None):

        super(CustomTensorboardCallback, self).on_epoch_end(epoch, logs)

        with self._writers['train'].as_default():
            if epoch % config.CHECKPOINT_WRITING_STEP == 0:
                if config.WRITE_IMAGES:
                    gt_image = self.gt_image
                    spectrum = self.spectrum

                    predictions = self.model.predict(np.expand_dims(spectrum, axis=-1))
                    image = tf.py_function(self.draw_predictions_on_images, [predictions], [tf.uint8])
                    tf.summary.image('image', image, step=epoch)
            
            tf.summary.scalar('epoch_specificity', data=logs['tn'] / (logs['tn'] + logs['fp']), step=epoch)
        
        with self._writers['val'].as_default():
            tf.summary.scalar('epoch_specificity', data=logs['val_tn'] / (logs['val_tn'] + logs['val_fp']), step=epoch)

        '''print(self.__dict__)
        for key, value in logs.items():
            print (key, value)'''