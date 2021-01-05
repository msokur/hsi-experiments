import config
from train import train
import numpy as np
from tqdm import tqdm
import glob
import test
import os
import csv

def dropout_experiment():
    for dropout in tqdm(np.linspace(0, 1, 10)):
        config.DROPOUT_VALUE = dropout
        old_model_name = config.MODEL_NAME
        config.MODEL_NAME += '_dropout_' + str(round(config.DROPOUT_VALUE, 2))

        train()

        config.MODEL_NAME = old_model_name

#name - of experiment, name of subfolder that will be created in "test"
#paths - paths with models that would be tested
def test_experiment(name, paths):
    SAVING_PATH = os.path.join('test', name)
    if not os.path.exists(SAVING_PATH):
        os.mkdir(SAVING_PATH)

    for path in paths:
        tester = test.Tester('logs', path.split('\\')[-1], 'cp-0050', ['data', 'test_test'], SAVING_PATH)
        tester.test_ALL_images(test_all_spectra=False, save=True, show=False, save_stats=True)

def cross_validation():
    if not os.path.exists(test.SAVING_PATH):
        os.mkdir(config.MODEL_NAME)
    config.MODEL_NAME = os.path.join(config.MODEL_NAME, config.MODEL_NAME.split('/')[-1])

    paths = glob.glob(os.path.join(config.DATA_PATHS[0], '*.dat'))
    for i, path in enumerate(paths):
        old_model_name = config.MODEL_NAME
        config.MODEL_NAME += path.split('/')[-1]

        model = train(paths=paths, except_indexes=[i])

        test.model = model
        test.test_one_image(path,
                            path_image=path + '_Mask JW Kolo.png',
                            save=False,
                            show=False,
                            test_all_spectra=False,
                            save_stats=True,
                            folder_name = config.MODEL_NAME)

        config.MODEL_NAME = old_model_name

if __name__ =='__main__':
    #cross_validation()

    paths = glob.glob('logs/test_inception*')
    test_experiment('dropout_experiment', paths)