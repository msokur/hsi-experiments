import datetime

DATA_PATHS = [r'C:\Users\Tkachenko\Desktop\HSI_data\data']
BATCH_SIZE = 2048
SPLIT_FACTOR = 0.9 #for data sets: train\test data percentage
WAVE_AREA = 100
FIRST_NM = 8
LAST_NM = 100
EPOCHS = 200
SCALER_FILE_NAME = '.scaler'

CHECKPOINT_PATH = 'checkpoints'
MODEL_PATH = 'model'
CHECKPOINT_WRITING_STEP = 5
WRITE_IMAGES = True

RESTORE_MODEL = True
MODEL_NAME = 'logs/' + 'inception_all_data_ill_weight_x2'

if not RESTORE_MODEL:
    current_time = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")
    MODEL_NAME += current_time
