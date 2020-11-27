from data_loader import *
import tensorflow as tf 
from tensorflow import keras
import config
from data_loader import get_data_for_showing
from model import *
from callbacks import CustomTensorboardCallback
import os


'''-------METRICS---------'''

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Recall(name='sensitivity'),
      keras.metrics.AUC(name='auc'),
]

'''-------TENSORBOARD---------'''
log_dir = config.MODEL_NAME
if not config.RESTORE_MODEL:
      os.mkdir(log_dir)

'''-------DATASET---------'''
train, test, class_weight = get_data(log_dir)

print(train[0], test[0])

'''-------MODEL---------'''

initial_epoch = 0
if config.RESTORE_MODEL:
      search_path = os.path.join(log_dir, 'checkpoints')
      all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
      sorted(all_checkpoints)
      all_checkpoints = np.array(all_checkpoints)

      initial_epoch = int(all_checkpoints[-1].split('-')[-1])

      model = tf.keras.models.load_model(all_checkpoints[-1])
else:
      model = inception1d_block()

      model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=METRICS)

model.summary()

'''-------CALLBACKS---------'''

tensorboard_callback = CustomTensorboardCallback(
      log_dir=log_dir, 
      histogram_freq=0, 
      write_graph=True, 
      write_images=True)

checkpoint_path = os.path.join(log_dir, config.CHECKPOINT_PATH, 'cp-{epoch:04d}') 

checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path, 
      verbose=1, 
      period = config.CHECKPOINT_WRITING_STEP)

'''-------TRAINING---------'''

model.fit(np.expand_dims(train[:, :-2], axis=-1),
    train[:, -2],
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    verbose=1,
    initial_epoch = initial_epoch,
    callbacks = [tensorboard_callback, checkpoints_callback],
    validation_data=(np.expand_dims(test[:, :-2], axis=-1), test[:, -2]),
    class_weight=class_weight, 
    sample_weight = train[:, -1])

    


