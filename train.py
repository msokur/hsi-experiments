import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, 'utils')
sys.path.insert(1, 'data_utils')
sys.path.insert(2, 'models')

import tensorflow as tf
from tensorflow import keras
import config
import model
import model_3d
import callbacks
import os
from shutil import copyfile
import telegram_send
from generator import DataGenerator
import tf_metrics
import numpy as np
import psutil

from utils import send_tg_message


def train_(log_dir, mirrored_strategy, paths=None, except_indexes=[]):
    '''-------LOGGING and HPARAMS---------'''
    if not config.RESTORE_MODEL:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        for file in config.FILES_TO_COPY:
            if os.path.exists(file):
                copyfile(file, os.path.join(log_dir, file.split(config.SYSTEM_PATHS_DELIMITER)[-1]))

    '''-------DATASET---------'''

    process = psutil.Process(os.getpid())

    # train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)
    train_generator = DataGenerator('train',
                                    config.SHUFFLED_PATH,
                                    config.BATCHED_PATH,
                                    log_dir,
                                    split_flag=True,
                                    except_indexes=except_indexes)
    valid_generator = DataGenerator('valid',
                                    config.SHUFFLED_PATH,
                                    config.BATCHED_PATH,
                                    log_dir,
                                    split_flag=False,
                                    except_indexes=except_indexes)
    class_weights = train_generator.get_class_weights()
    print(class_weights)


    def gen_train_generator():
        for i in range(train_generator.len):
            yield train_generator.getitem(i)

    train_dataset = tf.data.Dataset.from_generator(gen_train_generator, output_signature=config.OUTPUT_SIGNATURE)

    def gen_valid_generator():
        for i in range(valid_generator.len):
            yield valid_generator.getitem(i)

    valid_dataset = tf.data.Dataset.from_generator(gen_valid_generator, output_signature=config.OUTPUT_SIGNATURE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    valid_dataset = valid_dataset.with_options(options)

    '''-------MODEL---------'''

    initial_epoch = 0
    model = None
    METRICS = []
    WEIGTED_METRICS = []
    if config.RESTORE_MODEL:
        print('!!!!!!!!!!!!We restore model!!!!!!!!!!!!')
        search_path = os.path.join(log_dir, 'checkpoints')
        all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if
                           os.path.isdir(os.path.join(search_path, d))]
        sorted(all_checkpoints)
        all_checkpoints = np.array(all_checkpoints)

        initial_epoch = int(all_checkpoints[-1].split('-')[-1])

        if config.MODE == 'LOCAL' or config.MODE == 'SERVER':
            with mirrored_strategy.scope():
                model = keras.models.load_model(all_checkpoints[-1], custom_objects=config.CUSTOM_OBJECTS, compile=True)
        else:
            model = keras.models.load_model(all_checkpoints[-1], custom_objects=config.CUSTOM_OBJECTS)
    else:
        # model = model_3d.paper_model()
        model = model_3d.inception3d_model()
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Recall(name='sensitivity')
    ]
    WEIGHTED_METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        # keras.metrics.AUC(name='auc'),
        # tf_metrics.f1_m
        # specificity_m
    ]
    # model = inception_model()

    model.compile(
        # optimizer=keras.optimizers.Adam(lr=config.LEARNING_RATE, clipnorm=1.),
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        # optimizer=keras.optimizers.RMSprop(lr=config.LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=keras.metrics.BinaryAccuracy(name='accuracy'),
        # metrics=METRICS,
        # weighted_metrics=WEIGHTED_METRICS
    )

    model.summary()

    '''-------CALLBACKS---------'''

    tensorboard_callback = callbacks.CustomTensorboardCallback(
        log_dir=log_dir,
        # write_graph=True,
        # histogram_freq=1,
        # profile_batch = '20,30',
        except_indexes=except_indexes,
        train_generator=train_generator,
        strategy=mirrored_strategy,
        process=process)

    # gradient_callback = callbacks.GradientCallback()

    checkpoint_path = os.path.join(log_dir, config.CHECKPOINT_PATH, 'cp-{epoch:04d}')

    checkpoints_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=2,
        period=config.WRITE_CHECKPOINT_EVERY_Xth_STEP)

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        min_delta=0,
        patience=25,
        restore_best_weights=True)

    '''-------TRAINING---------'''

    callbacks_ = [tensorboard_callback, checkpoints_callback]
    if config.WITH_EARLY_STOPPING:
        callbacks_.append(early_stopping_callback)

    history = model.fit(
        # x=train_generator,
        # validation_data=valid_generator,
        x=train_dataset,
        validation_data=valid_dataset,
        epochs=config.EPOCHS,
        verbose=2,
        initial_epoch=initial_epoch,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks_,
        use_multiprocessing=True,
        class_weight=class_weights,
        workers=int(os.cpu_count()))

    np.save(os.path.join(log_dir, 'history.history'), history.history)

    return model, history


def train(paths=None, except_indexes=[]):
    history = None
    log_dir = config.MODEL_NAME
    model = None
    try:
        mirrored_strategy = None
        if config.MODE == 'LOCAL' or config.MODE == 'SERVER':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)

            # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
            # mirrored_strategy = tf.distribute.CentralStorageStrategy()
            mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
            # mirrored_strategy = tf.distribute.MirroredStrategy()

            with mirrored_strategy.scope():
                model, history = train_(log_dir, mirrored_strategy, paths=paths, except_indexes=except_indexes)
        else:
            model, history = train_(log_dir, mirrored_strategy, paths=paths, except_indexes=except_indexes)

    except Exception as e:
        last_epoch = -1
        if history is not None:
            last_epoch = len(history.history["loss"])

        send_tg_message(f'Mariia, ERROR!!!, training {log_dir} has finished after {last_epoch} epochs with error {e}')
        raise e  # TODO REMOVE!!

    checkpoints_paths = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoints_paths):
        os.mkdir(checkpoints_paths)

    final_model_save_path = os.path.join(log_dir, 'checkpoints', f'cp-{len(history.history["loss"]):04d}')
    if not os.path.exists(final_model_save_path):
        os.mkdir(final_model_save_path)
    model.save(final_model_save_path)

    if history is not None:
        send_tg_message(f'Mariia, training {log_dir} has finished after {len(history.history["loss"])} epochs')
    else:
        send_tg_message(f'Mariia, training {log_dir} has finished')

    return model


if __name__ == '__main__':
    train()
    #train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
