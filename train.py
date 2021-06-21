#import tensorflow
from tensorflow import keras
import config
from data_loader import get_data_for_showing
from data_loader import *
from model import *
from callbacks import CustomTensorboardCallback
import os
from shutil import copyfile
import telegram_send
from generator import DataGenerator

def send_tg_message(message):
    if config.MODE == 0:
            message = 'SERVER ' + message
    telegram_send.send(messages=[message], conf='tg.config')

def train(paths=None, except_indexes=[]):
    history = None
    log_dir = config.MODEL_NAME
    try:
        mirrored_strategy = None
        if config.MODE == 1:
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

            #mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            #mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
            mirrored_strategy = tf.distribute.experimental.CentralStorageStrategy()
            #mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        '''-------LOGGING and HPARAMS---------'''
        
        if not config.RESTORE_MODEL:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

            for file in ['config.py', 'train.py', 'data_loader.py', 'model.py']:
                copyfile(file, os.path.join(log_dir, file))
            job_filepath = 'start.job'
            if config.MODE == 0 and os.path.exists(job_filepath):
                copyfile(file, os.path.join(log_dir, job_filepath))

        with open(os.path.join(log_dir, 'comments.txt'),'a', newline='') as comments:
            comments.write(config.COMMENTS)

        '''-------DATASET---------'''
        #train, test, class_weight = get_data(log_dir, paths=paths, except_indexes=except_indexes)
        train_generator = DataGenerator('train', config.AUGMENTED_PATH, config.SHUFFLED_PATH, config.BATCHED_PATH, log_dir)
        valid_generator = DataGenerator('valid', config.AUGMENTED_PATH, config.SHUFFLED_PATH, config.BATCHED_PATH, log_dir)
        class_weights = train_generator.get_class_weights()
        
        #print(train[0], test[0])

        '''-------MODEL---------'''

        initial_epoch = 0
        model = None
        METRICS = []
        WEIGTED_METRICS = []
        if config.RESTORE_MODEL:
            search_path = os.path.join(log_dir, 'checkpoints')
            all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
            sorted(all_checkpoints)
            all_checkpoints = np.array(all_checkpoints)

            initial_epoch = int(all_checkpoints[-1].split('-')[-1])
            
            if config.MODE == 1:
                with mirrored_strategy.scope():
                    model = keras.models.load_model(all_checkpoints[-1])
            else:
                model = keras.models.load_model(all_checkpoints[-1]) 
        else:
            if config.MODE == 1:
                with mirrored_strategy.scope():
                    METRICS = [
                        keras.metrics.TruePositives(name='tp'),
                        keras.metrics.FalsePositives(name='fp'),
                        keras.metrics.TrueNegatives(name='tn'),
                        keras.metrics.FalseNegatives(name='fn'),
                        keras.metrics.Recall(name='sensitivity')
                    ]
                    WEIGHTED_METRICS = [
                        keras.metrics.BinaryAccuracy(name='accuracy'),
                        keras.metrics.AUC(name='auc')
                    ]
                    model = inception_model()
            else:
                METRICS = [
                    keras.metrics.TruePositives(name='tp'),
                    keras.metrics.FalsePositives(name='fp'),
                    keras.metrics.TrueNegatives(name='tn'),
                    keras.metrics.FalseNegatives(name='fn'),
                    keras.metrics.Recall(name='sensitivity')
                ]
                WEIGHTED_METRICS = [
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.AUC(name='auc')
                ]
                model = inception_model()
            #model = lstm_block()

            model.compile(
                optimizer=keras.optimizers.Adam(lr=config.LEARNING_RATE),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=METRICS,
                weighted_metrics=WEIGHTED_METRICS
                )

        model.summary()

        '''-------CALLBACKS---------'''

        tensorboard_callback = CustomTensorboardCallback(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=True,
                except_indexes=except_indexes)

        checkpoint_path = os.path.join(log_dir, config.CHECKPOINT_PATH, 'cp-{epoch:04d}')

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1,
                period=config.CHECKPOINT_WRITING_STEP)

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_auc',
            min_delta=0,
            patience=25,
            restore_best_weights=True)


        '''-------TRAINING---------'''

        callbacks = [tensorboard_callback, checkpoints_callback]
        if config.EARLY_STOPPING:
            callbacks.append(early_stopping_callback)

        #history = model.fit(np.expand_dims(train[:, :-2], axis=-1),
        '''history = model.fit(train[:, :-2],
            train[:, -2],
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            verbose=1,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            #validation_data=(np.expand_dims(test[:, :-2], axis=-1), test[:, -2], test[:, -1]),
            validation_data=(test[:, :-2], test[:, -2], test[:, -1]),
            class_weight=class_weight,
            sample_weight=train[:, -1])'''
        history = model.fit(x=train_generator,
                    validation_data=valid_generator,
                    epochs=config.EPOCHS,
                    verbose=1,
                    initial_epoch=initial_epoch,
                    batch_size=config.BATCH_SIZE,
                    callbacks=callbacks,
                    use_multiprocessing=True,
                    class_weight=class_weights)

        np.save(os.path.join(log_dir, '.history'), history.history)
    except Exception as e:
        print(e)
        raise e #TODO REMOVE!!

        if config.TELEGRAM_SENDING:
            last_epoch = -1
            if history is not None:
                last_epoch = len(history.history["loss"])
            
            send_tg_message(f'Mariia, ERROR!!!, training {log_dir} has finished after {last_epoch} epochs with error {e}')

    final_model_save_path = os.path.join(log_dir, 'checkpoints',  f'cp-{len(history.history["loss"]):04d}')
    if not os.path.exists(final_model_save_path):
        os.mkdir(final_model_save_path)
    model.save(final_model_save_path)

    if config.TELEGRAM_SENDING:
        send_tg_message(f'Mariia, training {log_dir} has finished after {len(history.history["loss"])} epochs')

    return model

if __name__ == '__main__':
    train()

    


