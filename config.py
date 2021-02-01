import datetime
import os

#change modelname here!
def get_model_name(MODEL_NAME_PATHS, model_name='lstm_inception_8'):
    return os.path.join(*MODEL_NAME_PATHS, model_name)

NORMALIZATION_TYPES = {
    'svn': 0,
    'l2_norm': 1
}

DATA_PATHS = [r'data']
BATCH_SIZE = 10000
SPLIT_FACTOR = 0.9 #for data sets: train\test data percentage
WAVE_AREA = 100
FIRST_NM = 8
LAST_NM = 100
EPOCHS = 500
CROSS_VALIDATION_SPLIT = int(44 / 4)
SCALER_FILE_NAME = '.scaler'
NORMALIZATION_TYPE = NORMALIZATION_TYPES['l2_norm']

CHECKPOINT_PATH = 'checkpoints'
MODEL_PATH = 'model'
CHECKPOINT_WRITING_STEP = 10
WRITE_IMAGES = True

DROPOUT_VALUE = 0.1

RESTORE_MODEL = True
ADD_TIME = False
MODEL_NAME_PATHS = ['logs']
MODEL_NAME = get_model_name(MODEL_NAME_PATHS)

COMMENTS = 'sample weight, dropout 0.2, fixed spectra, 46 files'

if not RESTORE_MODEL and ADD_TIME:
    current_time = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")
    MODEL_NAME += current_time
