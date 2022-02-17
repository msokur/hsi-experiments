from datetime import datetime
import os
import tensorflow as tf
import sys
import inspect
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(current_dir, 'utils'))
sys.path.insert(1, os.path.join(current_dir, 'data_utils'))
sys.path.insert(1, os.path.join(current_dir, os.path.join('data_utils', 'data_loaders')))
sys.path.insert(2, os.path.join(current_dir, 'models'))
sys.path.insert(3, os.path.join(current_dir, 'trainers'))

import tf_metrics

print('paths from config', sys.path)

# ----------------------------------------------------------------------------------------------------------

FILE_EXTENSIONS = {
    '_dat': '.dat',
    '_npz': '.npz',
    '_mat': '.mat'
}
FILE_EXTENSION = FILE_EXTENSIONS['_mat']

NORMALIZATION_TYPES = {
    'svn': 'svn',
    'l2_norm': 'l2_norm',
    'svn_T': 'svn_T'
}
NORMALIZATION_TYPE = NORMALIZATION_TYPES['svn_T']

DATABASES = {   # for data_loader
    'data_loader_easy': 'colon',
    'data_loader_mat_eso': 'bea_eso',
    'data_loader_mat_brain': 'bea_brain',
    'data_loader_mat_colon': 'bea_colon'
}


# ----------------------------------------------------------------------------------------------------------
bea_db = 'DatabaseBrainFMed'  # TODO remove

#RAW_NPZ_PATH = os.path.join('data_bea_db', bea_db, 'raw_3d_weighted')
RAW_NPZ_PATH = os.path.join('C:\\Users\\tkachenko\\Desktop\\HSI\\bea\\databases', bea_db, bea_db, 'raw_3d_weighted')

TEST_NPZ_PATH = os.path.join('C:\\Users\\tkachenko\\Desktop\\HSI\\bea\\databases', bea_db, bea_db)
SHUFFLED_PATH = os.path.join(RAW_NPZ_PATH, 'shuffled')
BATCHED_PATH = os.path.join(RAW_NPZ_PATH, 'batch_sized')

if 'Colon' in bea_db:
    DATABASE = DATABASES['data_loader_mat_colon']
elif 'Eso' in bea_db:
    DATABASE = DATABASES['data_loader_mat_eso']
elif 'Brain' in bea_db:
    DATABASE = DATABASES['data_loader_mat_brain']
else:
    DATABASE = DATABASES['data_loader_easy']

if 'bea' in DATABASE:
    FILE_EXTENSION = FILE_EXTENSIONS['_mat']

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
WITH_SAMPLE_WEIGHTS = True
WITH_BATCH_NORM = False
WITH_BACKGROUND_EXTRACTION = False
WITH_PREPROCESS_DURING_SPLITTING = False  # used in __split_arrays in preprocessor.py to run method preprocess()...
WITH_TUNING = True

WRITE_CHECKPOINT_EVERY_Xth_STEP = 2  # callbacks
WRITE_GRADIENTS_EVERY_Xth_BATCH = 50000000000  # callbacks
# every GRADIENTS_WRITING_STEP batches we write gradients, so not epochs - gradients
WRITE_IMAGES = False  # callbacks
WITH_EARLY_STOPPING = False  # callbacks

_3D = True  # data
_3D_SIZE = [5, 5]  # data
FIRST_NM = 8  # data
LAST_NM = 100  # data
SCALER_FILE_NAME = '.scaler'  # data
WAVE_AREA = 100  # data

INCEPTION_FACTOR = 8  # model
DROPOUT = 0.1  # model
NUMBER_OF_CLASSES_TO_TRAIN = 3  # model
if 'colon' in DATABASE:
    NUMBER_OF_CLASSES_TO_TRAIN = 2
if 'brain' in DATABASE:
    NUMBER_OF_CLASSES_TO_TRAIN = 4
LABELS_OF_CLASSES_TO_TRAIN = np.arange(NUMBER_OF_CLASSES_TO_TRAIN)  # data. It's possible that we have 4 classes in
# data, but want to use only 2 during training. If so we need to specify it here,
# for example, [1, 2] will mean that we will use only classes with labels 1 and 2.
# By default, we create labels with np.arange() corresponding to  NUMBER_OF_CLASSES_TO_TRAIN
# Reminder: you can set the labels itself overriding DataLoader.get_labels()

assert len(LABELS_OF_CLASSES_TO_TRAIN) == NUMBER_OF_CLASSES_TO_TRAIN  # check yourself

BATCH_SIZE = 100  # train
EPOCHS = 40  # train
LEARNING_RATE = 1e-4  # train

SPLIT_FACTOR = 0.9  # train   #for data sets: train\test data percentage

CV_HOW_MANY_PATIENTS_EXCLUDE = 1  # cv
HISTORY_ARGMIN = "val_loss"  # cv. through which parameter of history(returned by model.fit() and then saved) choose the
# best checkpoint in validation data

ADD_TIME = False  # pipeline   #whether to add time to logs paths
RESTORE_MODEL = False  # pipeline
TELEGRAM_SENDING = True  # utils
# ---------------------------------------Tuning params-----------------------------------------------


# General guide: https://keras.io/guides/keras_tuner/getting_started/#tune-model-training
TUNER_CLASS = 'BayesianOptimization'  # RandomSearch, BayesianOptimization or Hyperband. Read about differences:
# https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f
TUNER_MAX_TRIALS = 2
TUNER_EPOCHS = 1
TUNER_EPOCHS_PER_TRIAL = 1
TUNER_OBJECTIVE = "val_loss"  # Read about objective:
# https://keras.io/guides/keras_tuner/getting_started/#specify-the-tuning-objective
#TUNER_DIRECTORY = "tuner_results"
#TUNER_PROJECT_NAME = "inception_3d"
TUNER_ADD_TIME = True
TUNER_OVERWRITE = True
TUNER_MODEL = 'KerasTunerModelOnes'  # or KerasTunerModelOnes

# ---------------------------------------------------------------------------------------------------

# ----------------------------AUGMENTATION

AUGMENTATION = {
    'percent': 0.9,  # probability that value(feature) in sample will be augmented
    'range': [-0.02, 0.02],
    'new_rows_per_sample': 10
}

if WITH_AUGMENTATION:
    BATCH_SIZE = int(BATCH_SIZE / AUGMENTATION['new_rows_per_sample'])

# ----------------------------FILES_TO_COPY

import glob

files_to_copy = ['*.py', 
                 'data_utils/*.py', 
                 'models/*.py',
                 'data_utils/data_loaders/*.py',
                 'scrips/start_cv.job']

FILES_TO_COPY = []
for f in files_to_copy:
    FILES_TO_COPY += glob.glob(os.path.join(current_dir, f))

print('FILES_TO_COPY', FILES_TO_COPY)

# ----------------------------SYSTEM_PATHS_DELIMITER

import platform

if platform.system() == 'Windows':
    SYSTEM_PATHS_DELIMITER = '\\'
else:
    SYSTEM_PATHS_DELIMITER = '/'

# ----------------------------MODES
MODE_TYPES = {
    'CLUSTER': 'CLUSTER',
    'LOCAL_GPU': 'LOCAL_GPU',
    'LOCAL_NO_GPU': 'LOCAL_NO_GPU'
}

uname = platform.uname()

MODE = MODE_TYPES['CLUSTER']

if "clara" in uname.node:
    MODE = MODE_TYPES['CLUSTER']
if "scads" in uname.node:
    MODE = MODE_TYPES['LOCAL_NO_GPU']

# ----------------------------PATHS


def get_model_name(MODEL_NAME_PATHS, model_name='3d'):
    return os.path.join(*MODEL_NAME_PATHS, model_name)


if MODE == 'CLUSTER':
    prefix = r'/work/users/mi186veva/'

    RAW_NPZ_PATH = os.path.join(prefix, RAW_NPZ_PATH)
    SHUFFLED_PATH = os.path.join(prefix, SHUFFLED_PATH)
    BATCHED_PATH = os.path.join(prefix, BATCHED_PATH)
    TEST_NPZ_PATH = os.path.join(prefix, TEST_NPZ_PATH)

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

# ----------------------------CROSS_VALIDATION SPLIT

paths = glob.glob(os.path.join(RAW_NPZ_PATH, '*npz'))
CROSS_VALIDATION_SPLIT = int(len(paths) / CV_HOW_MANY_PATIENTS_EXCLUDE)  # int(number_of_all_patients / how_many_exclude_per_cv)

# ----------------------------OUTPUT FEATURES
OUTPUT_SIGNATURE_X_FEATURES = LAST_NM - FIRST_NM  # train
if DATABASE == 'bea_eso' or DATABASE == 'bea_colon':
    OUTPUT_SIGNATURE_X_FEATURES = 81
if DATABASE == 'bea_brain':
    OUTPUT_SIGNATURE_X_FEATURES = 128
    
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
