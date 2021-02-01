from data_loader import *
import tensorflow as tf 
from tensorflow import keras
import config
from data_loader import get_data_for_showing
from model import *
from callbacks import CustomTensorboardCallback
import os
from shutil import copyfile
import telegram_send


def train(paths=None, except_indexes=[-1]):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    #mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    #mirrored_strategy = tf.distribute.experimental.CentralStorageStrategy()
    #mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    '''-------LOGGING and HPARAMS---------'''
    log_dir = config.MODEL_NAME
    if not config.RESTORE_MODEL:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        for file in ['config.py', 'train.py', 'data_loader.py', 'model.py']:
            copyfile(file, os.path.join(log_dir, file))

    with open(os.path.join(log_dir, 'comments.txt'),'a', newline='') as comments:
        comments.write(config.COMMENTS)

    '''-------DATASET---------'''
    train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)

    print(train[0], test[0])

    '''-------MODEL---------'''

    initial_epoch = 0
    if config.RESTORE_MODEL:
          search_path = os.path.join(log_dir, 'checkpoints')
          all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
          sorted(all_checkpoints)
          all_checkpoints = np.array(all_checkpoints)

          initial_epoch = int(all_checkpoints[-1].split('-')[-1])

          with mirrored_strategy.scope():
              model = tf.keras.models.load_model(all_checkpoints[-1])
    else:
          with mirrored_strategy.scope():
              METRICS = [
                  keras.metrics.TruePositives(name='tp'),
                  keras.metrics.FalsePositives(name='fp'),
                  keras.metrics.TrueNegatives(name='tn'),
                  keras.metrics.FalseNegatives(name='fn'),
                  keras.metrics.BinaryAccuracy(name='accuracy'),
                  keras.metrics.Recall(name='sensitivity'),
                  keras.metrics.AUC(name='auc'),
              ]
              #model = inception_model()
              model = lstm_block()

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

    return model

if __name__ == '__main__':
    #telegram_send.configure("tg.config", group=True)
    #telegram_send.send(messages=['Hallo from mariias python'], conf='tg.config')
    train()

    


