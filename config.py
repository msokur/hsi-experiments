import datetime
import os

MODE_TYPES = {
    'SERVER':0,
    'LOCAL':1,
    'NO_GPU':2
}

MODE = MODE_TYPES['NO_GPU']

#change modelname here!
def get_model_name(MODEL_NAME_PATHS, model_name='inception_l2_norm_all_data'):
    return os.path.join(*MODEL_NAME_PATHS, model_name)

NORMALIZATION_TYPES = {
    'svn': 0,
    'l2_norm': 1
}
EARLY_STOPPING = False
INCEPTION_FACTOR = 16
TELEGRAM_SENDING = True

DATA_PATHS = []
if MODE == 0:
    DATA_PATHS = [r'/work/users/mi186veva/data', r'/work/users/mi186veva/data/data_additional']
else:
    DATA_PATHS = [r'data']#, r'data/data_additional']
    
DATA_LOADER_TYPES = {
    '_dat':0,
    '_npy':1
}


BATCH_SIZE = 10000
SPLIT_FACTOR = 0.9 #for data sets: train\test data percentage
WAVE_AREA = 100
FIRST_NM = 8
LAST_NM = 100
EPOCHS = 250
CROSS_VALIDATION_SPLIT = int(56 / 4)
SCALER_FILE_NAME = '.scaler'
NORMALIZATION_TYPE = NORMALIZATION_TYPES['l2_norm']

CHECKPOINT_PATH = 'checkpoints'
MODEL_PATH = 'model'
CHECKPOINT_WRITING_STEP = 50
WRITE_IMAGES = True

DROPOUT_VALUE = 0.1
LEARNING_RATE = 1e-3

RESTORE_MODEL = False
ADD_TIME = False

MODEL_NAME_PATHS = []
if MODE == 0:
    MODEL_NAME_PATHS = ['/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs']
else:
    MODEL_NAME_PATHS = ['logs'] 

MODEL_NAME = get_model_name(MODEL_NAME_PATHS)

COMMENTS = 'sample weight, dropout 0.2, fixed spectra, 46 files, weighted_metrics_weighted_val_data'

AUGMENTATION = {
    'augment': True,
    'percent': 0.9, #probability that value in sample will be augmented
    'range': [-0.02, 0.02],
    'new_rows_per_sample': 2
}

if not RESTORE_MODEL and ADD_TIME:
    current_time = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")
    MODEL_NAME += current_time
