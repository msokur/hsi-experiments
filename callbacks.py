import tensorflow as tf
# import cv2
# import data_loader
import config
import numpy as np
from tqdm import tqdm
import os
import math
import glob
import datetime
import test
from sklearn import preprocessing


# @tf.function(experimental_relax_shapes=True)
def distributed_train_step(strategy, func, batch, weights):
    per_replica_losses = strategy.run(func, args=(batch, weights))
    # return per_replica_losses
    # print(strategy.experimental_local_results(per_replica_losses))
    # print(strategy.experimental_local_results(per_replica_losses[0]).device)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=0)


class CustomTensorboardCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, except_indexes=[], train_generator=None, strategy=None, process=None, **kwargs):

        super(CustomTensorboardCallback, self).__init__(**kwargs)

        self.test_name = '2019_09_04_12_43_40_SpecCube.dat'
        # gt_image, spectrum_data, gesund_indexes, ill_indexes, not_certain_indexes = data_loader.get_data_for_showing(self.test_name, config.DATA_PATHS[0])
        # indexes = gesund_indexes + ill_indexes
        # indexes = np.array(indexes)

        # scaler = data_loader.restore_scaler(self.log_dir) #TODO return normal scaling
        self.scaler = preprocessing.Normalizer()
        # spectrum = spectrum_data[indexes[:, 0], indexes[:, 1]]
        # spectrum = self.scaler.transform(spectrum_data[indexes[:, 0], indexes[:, 1]])
        # self.gt_image = gt_image
        # self.gt = [0] * len(gesund_indexes) + [1] * len(ill_indexes)
        # self.spectrum = spectrum
        # self.indexes = indexes
        self.train_generator = train_generator
        self.strategy = strategy
        self.process = process

        self.are_excepted = False
        if len(except_indexes) > 0:
            self.are_excepted = True
            self.except_indexes = except_indexes
            # self.__get_spectrum_of_excluded_patients()

    def __get_spectrum_of_excluded_patients(self):
        self.excepted_spectrums = []
        self.excepted_gt = []

        for except_name in self.except_indexes:
            path = glob.glob(os.path.join(config.RAW_NPZ_PATH, except_name + '*'))

            if len(path) == 0:
                print(f'WARNING! For except_name {except_name} no raw_paths were found')
            else:
                data = np.load(path[0])
                not_not_certain_indexes = np.flatnonzero(data['y'] != 2)
                X = data['X'][not_not_certain_indexes]
                # X = self.scaler.transform(X[:, :-1])
                self.excepted_spectrums.append(X)
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
            image = image[image.shape[0] - shapes[0] - border:image.shape[0] - border, border:image.shape[1] - border,
                    :]

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
                    # image.itemset((indexes[counter, 0], indexes[counter, 1]), [255, 0, 0])
                elif key == 1:
                    image[indexes[counter, 0], indexes[counter, 1]] = [0, 255, 255]
                    # image.itemset((indexes[counter, 0], indexes[counter, 1]), [0, 255, 255])
                else:
                    image[indexes[counter, 0], indexes[counter, 1]] = [0, 0, 255]
                    # image.itemset((indexes[counter, 0], indexes[counter, 1]), [0, 0, 255])
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
        cv2.imwrite('test.png', image[..., ::-1])  # TODO delete
        if is_tf:
            image = image[..., ::-1]
        return image

    def __write_valid_scalar(self, scalar_name, scalar_value, epoch):
        if 'val' in self._writers:
            with self._writers['val'].as_default():
                tf.summary.scalar(scalar_name, data=scalar_value, step=epoch)
        if 'validation' in self._writers:
            with self._writers['validation'].as_default():
                tf.summary.scalar(scalar_name, data=scalar_value, step=epoch)

    def __count_grads(self, batch, weights):
        with tf.GradientTape() as g:
            data = np.load(self.train_generator.splitted_npz_paths[batch])
            X, y = data['X'], data['y']

            x = tf.convert_to_tensor(X[:, :-1], dtype=tf.float32)

            loss = self.model(x)  # calculate loss
            gradients = g.gradient(loss, weights)  # back-propagation

        return gradients

    '''def on_train_batch_end(self, batch, logs):
        if self.train_generator is None:
            return
        
        if batch % config.GRADIENTS_WRITING_STEP == 0:
            gradients = None
            weights = self.model.trainable_weights
            if (config.MODE == 1 or config.MODE == 0) and self.strategy is not None:
                with self.strategy.scope():
                    gradients = distributed_train_step(self.strategy, self.__count_grads, batch, weights)
            else:       
                gradients = self.__count_grads(batch, weights)

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(weights, gradients):
                with self._writers['train'].as_default():
                    tf.summary.histogram(weights.name.replace(':', '_') + '_grads', data=grads, step=self._epoch)'''

    def on_epoch_end(self, epoch, logs=None):

        super(CustomTensorboardCallback, self).on_epoch_end(epoch, logs)

        if self.process is not None:
            print("Memory : ", self.process.memory_info().rss)

        print('{0}, epoch {1} is ended'.format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), epoch + 1))

        with self._writers['train'].as_default():
            if epoch % config.WRITE_CHECKPOINT_EVERY_Xth_STEP == 0:
                if config.WRITE_IMAGES:
                    gt_image = self.gt_image
                    spectrum = self.spectrum

                    predictions = self.model.predict(np.expand_dims(spectrum,
                                                                    axis=-1))  # TODO don't forget to remove expand_dims (or better learn how to get the shape of an input layer)
                    image = tf.py_function(self.draw_predictions_on_images, [predictions], [tf.uint8])
                    tf.summary.image('image', image, step=epoch)

            # tf.summary.scalar('epoch_specificity', data=logs['tn'] / (logs['tn'] + logs['fp']), step=epoch)
            tf.summary.scalar('epoch_lr', data=self.model.optimizer.lr, step=epoch)

        # self.__write_valid_scalar('epoch_specificity', logs['val_tn'] / (logs['val_tn'] + logs['val_fp']), epoch)

        '''if self.are_excepted and epoch % config.CHECKPOINT_WRITING_STEP == 0:
            for name, exc, gt in zip(self.except_indexes, self.excepted_spectrums, self.excepted_gt):
                #predictions = self.model.predict(exc[:, :-1])
                predictions = self.model.predict(exc[::20])
                sensitivity, specificity, f1 = test.Tester.count_metrics(np.rint(gt)[::20], np.rint(predictions), "", "", False, return_dice=True)
                
                self.__write_valid_scalar('test_'+name+'_sensitivity', sensitivity, epoch)
                self.__write_valid_scalar('test_'+name+'_specificity', specificity, epoch)
                self.__write_valid_scalar('test_'+name+'_f1', f1, epoch)
        
                print(f'-------Epoch validation: {name} sensitivity:{sensitivity} specificity:{specificity} -----------')'''

        '''print('-------------LOGS----------------')
        for key, value in self.__dict__.items():   #callback items
            if key != 'gt':
                print (key, value)
        
        for key, value in logs.items():     #logs items
            print (key, value)'''
