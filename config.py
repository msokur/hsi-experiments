from datetime import datetime
import os
import tensorflow as tf
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(current_dir, 'utils'))
sys.path.insert(1, os.path.join(current_dir, 'data_utils'))
sys.path.insert(2, os.path.join(current_dir, 'models'))
sys.path.insert(3, os.path.join(current_dir, 'trainers'))

import tf_metrics

print('paths from config', sys.path)

# ----------------------------------------------------------------------------------------------------------

MODE_TYPES = {
    'SERVER': 'SERVER',
    'LOCAL': 'LOCAL',
    'NO_GPU': 'NO_GPU',
    'LOCAL_NO_GPU': 'LOCAL_NO_GPU'
}
MODE = MODE_TYPES['LOCAL_NO_GPU']

FILE_EXTENSIONS = {
    '_dat': '.dat',
    '_npz': '.npz',
    '_mat': '.mat'
}
FILE_EXTENSION = FILE_EXTENSIONS['_npz']

NORMALIZATION_TYPES = {
    'svn': 'svn',
    'l2_norm': 'l2_norm',
    'svn_T': 'svn_T'
}
NORMALIZATION_TYPE = NORMALIZATION_TYPES['l2_norm']

# ----------------------------------------------------------------------------------------------------------
RAW_NPZ_PATH = os.path.join('data_preprocessed', 'raw_3d_weighted')
SHUFFLED_PATH = os.path.join(RAW_NPZ_PATH, 'shuffled')
BATCHED_PATH = os.path.join(RAW_NPZ_PATH, 'batch_sized')

#SHUFFLED_PATH = r'data_preprocessed/augmented/shuffled'
#BATCHED_PATH = r'data_preprocessed/augmented/batch_sized'

#SHUFFLED_PATH = r'data_preprocessed/combi/shuffled'
#BATCHED_PATH = r'data_preprocessed/combi/batch_sized'

#SHUFFLED_PATH = r'data_preprocessed/combi_with_raw_ill/shuffled'
#BATCHED_PATH = r'data_preprocessed/combi_with_raw_ill/batch_sized'

#SHUFFLED_PATH = r'data_bea/ColonData/raw_3d_weights/shuffled'
#BATCHED_PATH = r'data_bea/ColonData/raw_3d_weights/batch_sized'

#SHUFFLED_PATH = r'data_preprocessed/augmented_l2_norm/shuffled'
#BATCHED_PATH = r'data_preprocessed/augmented_l2_norm/batch_sized'

CHECKPOINT_PATH = 'checkpoints'
MODEL_PATH = 'model'
# ----------------------------------------------------------------------------------------------------------

WITH_AUGMENTATION = False
WITH_NOT_CERTAIN = False
WITH_SAMPLE_WEIGHTS = False
WITH_BATCH_NORM = False
WITH_BACKGROUND_EXTRACTION = False
WITH_PREPROCESS_DURING_SPLITTING = False  # used in __split_arrays in preprocessor.py to run method preprocess()...
WITH_TUNING = False

WRITE_CHECKPOINT_EVERY_Xth_STEP = 2  # callbacks
WRITE_GRADIENTS_EVERY_Xth_BATCH = 50000000000  # callbacks
# every GRADIENTS_WRITING_STEP batches we write gradients, so not epochs - gradients
WRITE_IMAGES = False  # callbacks
WITH_EARLY_STOPPING = False  # callbacks

_3D_SIZE = [5, 5]  # data
FIRST_NM = 8  # data
LAST_NM = 100  # data
SCALER_FILE_NAME = '.scaler'  # data
WAVE_AREA = 100  # data

INCEPTION_FACTOR = 8  # model
DROPOUT = 0.1  # model

BATCH_SIZE = 100  # train
EPOCHS = 20  # train
LEARNING_RATE = 1e-4  # train
OUTPUT_SIGNATURE_X_FEATURES = LAST_NM - FIRST_NM  # train
SPLIT_FACTOR = 0.9  # train   #for data sets: train\test data percentage

CROSS_VALIDATION_SPLIT = int(56 / 1)  # cv   #int(number_of_all_patients / how_many_exclude_per_cv)

ADD_TIME = True  # pipeline   #whether to add time to logs paths
RESTORE_MODEL = False  # pipeline
TELEGRAM_SENDING = True  # utils
# ---------------------------------------Tuning params-----------------------------------------------


# General guide: https://keras.io/guides/keras_tuner/getting_started/#tune-model-training
TUNER_CLASS = 'BayesianOptimization'  # RandomSearch, BayesianOptimization or Hyperband. Read about differences:
# https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f
TUNER_MAX_TRIALS = 1
TUNER_EPOCHS = 1
TUNER_EPOCHS_PER_TRIAL = 1
TUNER_OBJECTIVE = "val_loss"  # Read about objective:
# https://keras.io/guides/keras_tuner/getting_started/#specify-the-tuning-objective
TUNER_DIRECTORY = "tuner_results"
TUNER_PROJECT_NAME = "inception_3d"
TUNER_ADD_TIME = True
TUNER_OVERWRITE = True

# ---------------------------------------------------------------------------------------------------

# ----------------------------OUTPUT_SIGNATURE
if WITH_SAMPLE_WEIGHTS:
    OUTPUT_SIGNATURE = (
        tf.TensorSpec(shape=(None, _3D_SIZE[0], _3D_SIZE[1], OUTPUT_SIGNATURE_X_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32))
else:
    OUTPUT_SIGNATURE = (
        tf.TensorSpec(shape=(None, _3D_SIZE[0], _3D_SIZE[1], OUTPUT_SIGNATURE_X_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32))

# ----------------------------AUGMENTATION

AUGMENTATION = {
    'percent': 0.9,  # probability that value(feature) in sample will be augmented
    'range': [-0.02, 0.02],
    'new_rows_per_sample': 10
}

if WITH_AUGMENTATION:
    BATCH_SIZE = int(BATCH_SIZE / AUGMENTATION['new_rows_per_sample'])
    print(BATCH_SIZE)

# ----------------------------FILES_TO_COPY

import glob

FILES_TO_COPY = glob.glob('*.py')
FILES_TO_COPY += glob.glob('data_utils/*.py')
FILES_TO_COPY += glob.glob('models/*.py')
FILES_TO_COPY += ['scrips/start_cv.job']

# ----------------------------SYSTEM_PATHS_DELIMITER

import platform

if platform.system() == 'Windows':
    SYSTEM_PATHS_DELIMITER = '\\'
else:
    SYSTEM_PATHS_DELIMITER = '/'


# ----------------------------PATHS

def get_model_name(MODEL_NAME_PATHS, model_name='3d'):
    return os.path.join(*MODEL_NAME_PATHS, model_name)


if MODE == 'SERVER' or MODE == 'NO_GPU':
    prefix = r'/work/users/mi186veva/'

    RAW_NPZ_PATH = os.path.join(prefix, RAW_NPZ_PATH)
    SHUFFLED_PATH = os.path.join(prefix, SHUFFLED_PATH)
    BATCHED_PATH = os.path.join(prefix, BATCHED_PATH)

    # MODEL_NAME_PATHS = ['/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs', 'debug_combi']
    MODEL_NAME_PATHS = ['/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs']
else:
    MODEL_NAME_PATHS = ['logs']

MODEL_NAME = get_model_name(MODEL_NAME_PATHS)

pth = ''
for path_part in MODEL_NAME_PATHS:
    pth = os.path.join(pth, path_part)
    if not os.path.exists(pth):
        os.mkdir(pth)

if not RESTORE_MODEL and ADD_TIME:
    MODEL_NAME += datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

# ----------------------------CUSTOM_OBJECTS

CUSTOM_OBJECTS = {'f1_m': tf_metrics.f1_m}
