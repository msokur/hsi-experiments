import datetime
import os
import tf_metrics

MODE_TYPES = {
    'SERVER':0,
    'LOCAL':1,
    'NO_GPU':2
}

MODE = MODE_TYPES['SERVER']

#change modelname here!
#def get_model_name(MODEL_NAME_PATHS, model_name='combi_with_raw_all'):
#def get_model_name(MODEL_NAME_PATHS, model_name='combi_min_max_WRA_model_50max_4inc'):
def get_model_name(MODEL_NAME_PATHS, model_name='combi_bg'):
    return os.path.join(*MODEL_NAME_PATHS, model_name)

NORMALIZATION_TYPES = {
    'svn': 0,
    'l2_norm': 1
}
EARLY_STOPPING = False
INCEPTION_FACTOR = 8
TELEGRAM_SENDING = True

DATA_PATHS = [r'data', r'data/data_additional'] #for data loader without generator
NPY_PATHS = [r'data_preprocessed/augmented'] #for data loader without generator
RAW_NPY_PATH = r'data_preprocessed/raw' #for generators
AUGMENTED_PATH = r'data_preprocessed/augmented' #for generators
#SHUFFLED_PATH = r'data_preprocessed/augmented/shuffled' #for generators
#SHUFFLED_PATH = r'data_preprocessed/augmented/shuffled'
#BATCHED_PATH = r'data_preprocessed/augmented/batch_sized' #for generators

#SHUFFLED_PATH = r'data_preprocessed/combi/shuffled'
#BATCHED_PATH = r'data_preprocessed/combi/batch_sized'

SHUFFLED_PATH = r'data_preprocessed/combi_with_raw_ill/shuffled'
BATCHED_PATH = r'data_preprocessed/combi_with_raw_ill/batch_sized'

#SHUFFLED_PATH = r'data_preprocessed/augmented_l2_norm/shuffled'
#BATCHED_PATH = r'data_preprocessed/augmented_l2_norm/batch_sized'

MODEL_NAME_PATHS = []
if MODE == 0 or MODE == 2:
    add_path = r'/work/users/mi186veva/'
    DATA_PATHS = [os.path.join(add_path, i) for i in DATA_PATHS]
    NPY_PATHS = [os.path.join(add_path, i) for i in NPY_PATHS]
    
    RAW_NPY_PATH = os.path.join(add_path, RAW_NPY_PATH)
    AUGMENTED_PATH = os.path.join(add_path, AUGMENTED_PATH)
    SHUFFLED_PATH = os.path.join(add_path, SHUFFLED_PATH)
    BATCHED_PATH = os.path.join(add_path, BATCHED_PATH)

    #MODEL_NAME_PATHS = ['/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs', 'debug_combi']
    MODEL_NAME_PATHS = ['/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs']
else:
    MODEL_NAME_PATHS = ['logs'] 
    
DATA_LOADER_TYPES = {
    '_dat':0,
    '_npy':1
}
DATA_LOADER_MODE = DATA_LOADER_TYPES['_dat']
NOT_CERTAIN_FLAG = False

BATCH_SIZE = 10000
SPLIT_FACTOR = 0.9 #for data sets: train\test data percentage
WAVE_AREA = 100
FIRST_NM = 8
LAST_NM = 100

EPOCHS = 20
CHECKPOINT_WRITING_STEP = 2
GRADIENTS_WRITING_STEP = 50000000000 #every GRADIENTS_WRITING_STEP batches we write gradients, so not epochs - gradients

CROSS_VALIDATION_SPLIT = int(56 / 1)
SCALER_FILE_NAME = '.scaler'
NORMALIZATION_TYPE = NORMALIZATION_TYPES['l2_norm']
WITH_BATCH_NORM = False
WITH_BACKGROUND_EXTRACTION = True
READING_TEST_DATA_FROM_DAT = False #other oprion is from .npz
CUSTOM_OBJECTS = {'f1_m':tf_metrics.f1_m}

CHECKPOINT_PATH = 'checkpoints'
MODEL_PATH = 'model'

WRITE_IMAGES = False
DROPOUT_VALUE = 0.1
LEARNING_RATE = 1e-4

RESTORE_MODEL = False
ADD_TIME = True



MODEL_NAME = get_model_name(MODEL_NAME_PATHS)

COMMENTS = 'sample weight, dropout 0.2, fixed spectra, 46 files, weighted_metrics_weighted_val_data'

AUGMENTATION = {
    'augment': False,
    'percent': 0.9, #probability that value in sample will be augmented
    'range': [-0.02, 0.02],
    'new_rows_per_sample': 10
}

if AUGMENTATION['augment']:
    BATCH_SIZE = int(BATCH_SIZE / AUGMENTATION['new_rows_per_sample'])
    print(BATCH_SIZE)

if not RESTORE_MODEL and ADD_TIME:
    current_time = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")
    MODEL_NAME += current_time

pth = ''
for path_part in MODEL_NAME_PATHS:
    pth = os.path.join(pth, path_part)
    if not os.path.exists(pth):
        os.mkdir(pth)
    