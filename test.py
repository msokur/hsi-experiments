import tensorflow as tf 
import cv2
import numpy as np
import os
from tqdm import tqdm
from data_loader import *
from callbacks import CustomTensorboardCallback


FOLDER_NAME = os.path.join(r'C:\Users\Tkachenko\Desktop\HSI_data\logs', 'inception_all_data_ill_weight_x2\checkpoints')
MODEL_PATH = os.path.join(FOLDER_NAME, 'cp-0150')
TEST_PATHS = ['data']
SAVING_PATH = 'test'

#os.mkdir()

model = tf.keras.models.load_model(MODEL_PATH)
tensorboard_callback = CustomTensorboardCallback(log_dir=r'C:\Users\Tkachenko\Desktop\HSI_data\logs\inception_all_data_ill_weight_x2') #for drawing function

scaler = restore_scaler(r'C:\Users\Tkachenko\Desktop\HSI_data\logs\inception_all_data_ill_weight_x2')

def test_one_image(path_dat, path_image=None, save=False, show=True, test_all_spectra = False):
    gt_image, spectrum_data, indexes = get_data_for_showing(path_dat, "")

    print('gt_image.shape', gt_image.shape)

    
    #scaler = restore_scaler(FOLDER_NAME)

    if not test_all_spectra:
        spectrum = scaler.transform(spectrum_data[indexes[:, 1], gt_image.shape[0] - indexes[:, 0] - 1])
    else:
        spectrum = scaler.transform(spectrum_data.reshape(spectrum_data.shape[0] * spectrum_data.shape[1], spectrum_data.shape[2]))
        indexes_0 = []
        indexes_1 = []
        for i in range(gt_image.shape[1]):
            indexes_0.append([i] * gt_image.shape[0])
            indexes_1.append(np.arange(gt_image.shape[0]))
        indexes = np.array([np.array(indexes_1).flatten(), np.array(indexes_0).flatten()]).T
        print(indexes.shape)


    predictions = model.predict(np.expand_dims(spectrum, axis=-1))


    tensorboard_callback.gt_image = gt_image[..., ::-1]
    tensorboard_callback.spectrum = spectrum
    tensorboard_callback.indexes = indexes

    result_image = []
    if path_image == None:
        result_image = tensorboard_callback.draw_predictions_on_images(predictions, image=None)
    else:
        image = cv2.imread(path_image)
        result_image = tensorboard_callback.draw_predictions_on_images(predictions, image=image)

    result_image = result_image[..., ::-1]

    if save:
        name = path_dat.split('\\')[-1].split('_S')[0]
        cv2.imwrite(os.path.join(SAVING_PATH, name + '.png'), result_image)

    if show:
        cv2.imshow('frame', result_image)
        cv2.waitKey(0)

def test_ALL_images(save=True, show=False, test_all_spectra = False):

    for path_dir in TEST_PATHS:
        for path in tqdm(glob.glob(os.path.join(path_dir, '*.dat'))):
            with open(path, newline='') as filex:
                filename=filex.name
                
                test_one_image(filename, save=save, show=show, test_all_spectra=test_all_spectra)
                


if __name__ == "__main__":
    test_one_image(r'data/2019_10_30_14_30_27_SpecCube.dat', path_image=r'data\2019_10_30_14_30_27_SpecCube.dat_Mask JW Kolo.png', save=False, show=True, test_all_spectra=True)
    #test_ALL_images( test_all_spectra = False)