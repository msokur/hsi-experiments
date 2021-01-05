import datetime

DATA_PATHS = [r'data']
BATCH_SIZE = 10000
SPLIT_FACTOR = 0.9 #for data sets: train\test data percentage
WAVE_AREA = 100
FIRST_NM = 8
LAST_NM = 100
EPOCHS = 50
SCALER_FILE_NAME = '.scaler'

CHECKPOINT_PATH = 'checkpoints'
MODEL_PATH = 'model'
CHECKPOINT_WRITING_STEP = 25
WRITE_IMAGES = True

DROPOUT_VALUE = 0.1

RESTORE_MODEL = False
MODEL_NAME = 'logs/' + 'cv_inception_sample_weight'

COMMENTS = 'sample weight, dropout 0.2, fixed spectra, 46 files'

'''if not RESTORE_MODEL:
    current_time = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")
    MODEL_NAME += current_time'''
